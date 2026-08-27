#pragma once
#include "numeric_types.h"
#include "splitkv.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename accumType, typename reduceType, const int SPLIT_COUNT, const bool UnRoll, const bool Tail, const int kHeadDim, typename Params>
__global__ void flash_fwd_splitkv_reduce_kernel(
        Params params) {
    static_assert(SPLIT_COUNT <= 64 and (kHeadDim % 128 == 0 or kHeadDim == 64));
    int num_splits = UnRoll ? SPLIT_COUNT: params.num_splits;
    float* scores_max_ptr = params.scores_max_ptr;
    float* scores_sum_ptr = params.scores_sum_ptr;

    // 128 threads, each thread won't process more than 4 half data, 2 is appropriate, for 64 threads to processing 128 half
    __shared__ float lds[512];
    int tx               = threadIdx.x;
    int block_x          = blockIdx.x;
    int s_m_split_stride = gridDim.x; // offset from the next split

    // recompute the true actual_seqlen_k and num_split
    const int bidb = block_x / (params.h * params.seqlen_q);
    int actual_seqlen_k;
    if (params.is_seqlens_k_cumulative) {
        actual_seqlen_k = params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb];
    } else {
        actual_seqlen_k = params.cu_seqlens_k[bidb];
    }

    // compute partition_size when fix num_splits
    int partition_size = params.partition_size > MLA_MAX_SPLITS ? splitkv_get_partitionsize_of_fix_numsplits(actual_seqlen_k, params.num_splits): params.partition_size;
    const int true_num_splits = Tail ? max(1, floor_div(actual_seqlen_k, partition_size)): ceil_div(actual_seqlen_k, partition_size);
    // const int true_num_splits = num_splits;

    bool exceed_split = (tx >= true_num_splits); // process boundary

    // each thread dispatch 1 piece of buffer_load; outliners will be assigned minimum value
    float s_max_load_ori = exceed_split ? -INFINITY: scores_max_ptr[block_x + tx * s_m_split_stride];
    // in a warp, reduce a true max value among 64 threads
    float s_max_tmp = s_max_load_ori;
    #pragma unroll
    for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
        s_max_tmp = max(s_max_tmp, __shfl_xor_tmp(s_max_tmp, step));
    }
    // compute rescale coefficient for max (numerator)
    float s_max_ratio = __expf(s_max_load_ori - s_max_tmp);

    // as above, reduce a true sum value amoing 64 threads in each wave
    float s_sum_load_ori = exceed_split ? 0.f: scores_sum_ptr[block_x + tx * s_m_split_stride];
    float s_sum_tmp = s_sum_load_ori * s_max_ratio;
    #pragma unroll
    for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
        s_sum_tmp = s_sum_tmp + __shfl_xor_tmp(s_sum_tmp, step);
    }

    // max-rescale coefficient x sum-rescale coefficient
    lds[tx] = s_sum_load_ori * s_max_ratio / s_sum_tmp;

    // finally, do rescale for each split and reduce the sum of them
    // each block(1waves) process (num_splits x head_dim) elements in total
    // for head_dim 128, each thread process 2 halfs for num_splits times
    constexpr int tx_float_count = kHeadDim >> 6;
    float tx_accum[tx_float_count] = {0.f};
    // offset from the next split for output from previous kernel, split * (batch, head,seq) * headdim
    int oaccum_stride = s_m_split_stride * kHeadDim;
    // int tx_offset= block_x * kHeadDim + tx * tx_float_count;
    int in_batch_offset = block_x - bidb * params.h * params.seqlen_q;
    int bidh = in_batch_offset / params.seqlen_q;
    int bids = in_batch_offset - bidh * params.seqlen_q;
    int real_block_x = params.layout == 0 ? block_x/*bhsd layout*/: bidb * params.seqlen_q * params.h + bids * params.h + bidh/*bshd layout*/;
    int tx_offset = real_block_x * kHeadDim + (tx & 63) * tx_float_count;
    reduceType* output_ptr = reinterpret_cast<reduceType*>(params.o_ptr) + tx_offset;
    accumType* oaccum_ptr  = reinterpret_cast<accumType*>(params.oaccum_ptr);
    // num_splits may not be 64, and thus need boundary judgement
    for (int i = 0; i < num_splits; ++i) {
        // read ultimate scale value for current split
        float s_scale      = lds[i];
        bool within_splits = (i < true_num_splits);
        for (int t = 0; t < tx_float_count; t += 2) {
            if constexpr (kHeadDim % 128 == 0) {
                // read ultimate scale value for current split
                vec2_Element<accumType> load = *(vec2_Element<accumType>*)(oaccum_ptr + tx_offset + t);
                // half -> float32, reduce precision loss
                float a_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[0]): 0.f;
                float b_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[1]): 0.f;
                // do rescale and sum
                tx_accum[t]     = __llvm_fma_f32(a_f32, s_scale, tx_accum[t]);
                tx_accum[t + 1] = __llvm_fma_f32(b_f32, s_scale, tx_accum[t + 1]);
            } else if constexpr (kHeadDim == 64) {
                // read ultimate scale value for current split
                accumType load  = *(accumType*)(oaccum_ptr + tx_offset + t);
                // half -> float32, reduce precision loss
                float load_f32  = within_splits? splitkv_upcast_to_f32<accumType>(load): 0.f;
                // do rescale and sum
                tx_accum[t]     = __llvm_fma_f32(load_f32, s_scale, tx_accum[t]);
            }
        }
        // switch to next split
        tx_offset += oaccum_stride;
    }
    // write results
    for (int t = 0; t < tx_float_count; t += 2) {
        if constexpr (kHeadDim % 128 == 0) {
            vec2_Element<reduceType> accum_result;
            #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                accum_result = DownCastPairNoPack<float, reduceType>(tx_accum[t], tx_accum[t + 1]);
            #else
                accum_result[0] = DownCast<float, reduceType, true>(tx_accum[t]);
                accum_result[1] = DownCast<float, reduceType, true>(tx_accum[t + 1]); // here, v_cvt_pkrtz can be used
            #endif
            *(vec2_Element<reduceType>*)(output_ptr + t) = accum_result;
        } else if constexpr (kHeadDim == 64) {
            reduceType accum_result = DownCast<float, reduceType, false>(tx_accum[t]);
            output_ptr[t] = accum_result;
        }
    }
}


template<typename accumType, typename reduceType, const int SPLIT_COUNT, const bool UnRoll, const bool Tail, const int kHeadDim, typename Params>
__global__ void flash_fwd_splitkv_reduce_kernel_split128(Params params) {

    static_assert(SPLIT_COUNT % 128 == 0 and (kHeadDim % 128 == 0 or kHeadDim == 64));
    constexpr int WAVES_COUNT = SPLIT_COUNT >> 6;
    int num_splits = UnRoll ? SPLIT_COUNT: params.num_splits;
    float* scores_max_ptr = params.scores_max_ptr;
    float* scores_sum_ptr = params.scores_sum_ptr;

    // each workgroup need SPLIT_COUNT threads to process SPLIT_COUNT num_splits, each thread process (kHeadDim / 64) floats for final accumulation
    constexpr int LDS_ACCUM = (SPLIT_COUNT * (kHeadDim >> 6));
    // prepare workspace of 2 floats to reduce max/sum in 2 waves
    constexpr int LDS_SIZE  = LDS_ACCUM + (SPLIT_COUNT >> 6);
    static_assert (LDS_SIZE * sizeof(float) <= 64 * 1024 and "Exceed max lds usage!");
    __shared__ float lds[LDS_SIZE];
    int tx               = threadIdx.x;
    int block_x          = blockIdx.x;
    int s_m_split_stride = gridDim.x; // offset from the next split

    // recompute the true actual_seqlen_k and num_split
    const int bidb = block_x / (params.h * params.seqlen_q);
    int actual_seqlen_k;
    if (params.is_seqlens_k_cumulative) {
        actual_seqlen_k = params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb];
    } else {
        actual_seqlen_k = params.cu_seqlens_k[bidb];
    }
    // compute partition_size when fix num_splits
    int partition_size = params.partition_size > MLA_MAX_SPLITS ? splitkv_get_partitionsize_of_fix_numsplits(actual_seqlen_k, params.num_splits): params.partition_size;
    const int true_num_splits = Tail ? max(1, floor_div(actual_seqlen_k, partition_size)): ceil_div(actual_seqlen_k, partition_size);
    // const int true_num_splits = num_splits;

    bool exceed_split = (tx >= true_num_splits); // process boundary

    // each thread dispatch 1 piece of buffer_load; outliners will be assigned minimum value
    float s_max_load_ori = exceed_split ? -INFINITY: scores_max_ptr[block_x + tx * s_m_split_stride];
    // in a warp, reduce a true max value among 64 threads
    float s_max_tmp = s_max_load_ori;
    #pragma unroll
    for (int step = 64 >> 1; step > 0; step = (step >> 1)) {
        s_max_tmp = max(s_max_tmp, __shfl_xor_tmp(s_max_tmp, step));
    }
    // for multiple waves, store the reduced max value to lds individually, and recompute max across multiple waves
    int wave_id = (tx >> 6);
    lds[LDS_ACCUM + wave_id] = s_max_tmp;
    __syncthreads();
    float lds_accum_temp = lds[LDS_ACCUM];
    #pragma unroll
    for (int s = 1; s < WAVES_COUNT; ++s) {
        lds_accum_temp = max(lds_accum_temp, lds[LDS_ACCUM + s]);
    }
    lds[LDS_ACCUM] = lds_accum_temp;
    __syncthreads();
    // acquire the reduced max value across multiple waves
    s_max_tmp = lds[LDS_ACCUM];

    // compute rescale coefficient for max (numerator)
    float s_max_ratio = __expf(s_max_load_ori - s_max_tmp);

    // as above, reduce a true sum value amoing 64 threads in each wave
    float s_sum_load_ori = exceed_split ? 0.f: scores_sum_ptr[block_x + tx * s_m_split_stride];
    float s_sum_tmp = s_sum_load_ori * s_max_ratio;
    #pragma unroll
    for (int step = 64 >> 1; step > 0; step = (step >> 1)) {
        s_sum_tmp = s_sum_tmp + __shfl_xor_tmp(s_sum_tmp, step);
    }
    // for multiple wave, store the reduced sum value to lds individually, and recompute sum across multiple waves
    lds[LDS_ACCUM + wave_id] = s_sum_tmp;
    __syncthreads();
    lds_accum_temp = lds[LDS_ACCUM];
    #pragma unroll
    for (int s = 1; s < WAVES_COUNT; ++s) {
        lds_accum_temp = lds_accum_temp + lds[LDS_ACCUM + s];
    }
    lds[LDS_ACCUM] = lds_accum_temp;
    __syncthreads();
    s_sum_tmp = lds[LDS_ACCUM];

    // max-rescale coefficient x sum-rescale coefficient
    lds[tx] = s_sum_load_ori * s_max_ratio / s_sum_tmp;

    // finally, do rescale for each split and reduce the sum of them
    // each block(multiple waves) process (num_splits x head_dim) elements in total
    // e.g. for head_dim 128, each thread process 2 halfs for num_splits times
    // e.g. for head_dim 512, each thread process 8 halfs for num_splits times
    constexpr int tx_float_count = kHeadDim >> 6;
    float tx_accum[tx_float_count] = {0.f};
    static_assert (tx_float_count * 128 < LDS_SIZE && "for each thread, it's not allowed to processing more than 8 half data");
    // offset from the next split for output from previous kernel, split * (batch, head,seq) * headdim
    int  oaccum_stride = s_m_split_stride * kHeadDim;
    // each wave read data from 0 in 128 halfs, and thus (tx % 64)
    // int  tx_offset = block_x * kHeadDim + (tx & 63) * tx_float_count;
    int in_batch_offset = block_x - bidb * params.h * params.seqlen_q;
    int bidh = in_batch_offset / params.seqlen_q;
    int bids = in_batch_offset - bidh * params.seqlen_q;
    int real_block_x = params.layout == 0 ? block_x/*bhsd layout*/: bidb * params.seqlen_q * params.h + bids * params.h + bidh/*bshd layout*/;
    int tx_offset = real_block_x * kHeadDim + (tx & 63) * tx_float_count;
    int  begin     = wave_id << 6;
    reduceType* output_ptr = reinterpret_cast<reduceType*>(params.o_ptr) + tx_offset;
    // for wave 0, splits [0, 63]; for wave 1, splits [64, 127]; for wave 2, splits [128, 191] ......
    accumType* oaccum_ptr = reinterpret_cast<accumType*>(params.oaccum_ptr) + begin * oaccum_stride;
    // num_splits may not be multiple of 64, and thus, the multiple waves need boundary judgement
    int split_count_this_wave = UnRoll ? 64: min(64, num_splits - begin);
    for (int i = 0; i < split_count_this_wave; ++i) {
        // read ultimate scale value for current split
        float s_scale      = lds[begin + i];
        bool within_splits = (begin + i) < true_num_splits;
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            if constexpr (kHeadDim % 128 == 0) {
                // read 2 halfs from current split of this threads
                vec2_Element<accumType> load = *(vec2_Element<accumType>*)(oaccum_ptr + tx_offset + t);
                // half -> float32, reduce precision loss
                float a_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[0]): 0.f;
                float b_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[1]): 0.f;
                // do rescale and sum
                tx_accum[t]     = __llvm_fma_f32(a_f32, s_scale, tx_accum[t]);
                tx_accum[t + 1] = __llvm_fma_f32(b_f32, s_scale, tx_accum[t + 1]);
            } else if constexpr (kHeadDim == 64) {
                // read 1 half from current split of this threads
                accumType load = *(accumType*)(oaccum_ptr + tx_offset + t);
                // half -> float32, reduce precision loss
                float load_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load): 0.f;
                // do rescale and sum
                tx_accum[t]     = __llvm_fma_f32(load_f32, s_scale, tx_accum[t]);
            }
        }
        // switch to next split
        tx_offset += oaccum_stride;
    }
    // no ds_read op again
    __syncthreads();
    // for multiple waves, store sum value to lds
    #pragma unroll
    for (int t = 0; t < tx_float_count; t += 2) {
        lds[tx * tx_float_count + t]     = tx_accum[t];
        if constexpr (kHeadDim % 128 == 0) {
            lds[tx * tx_float_count + t + 1] = tx_accum[t + 1];
        }
    }
    __syncthreads();
    // the 0th wave does the reduction and write operations
    if (wave_id == 0) {
        using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            if constexpr (kHeadDim % 128 == 0) {
                vec2_fp32 this_wave_f32s = *(vec2_fp32*)(lds + tx * tx_float_count + t);
                #pragma unroll
                for (int s = 1; s < (SPLIT_COUNT >> 6); ++s) {  // 0 wave accumulate data from other waves
                    vec2_fp32 other_wave_f32s = *(vec2_fp32*)(lds + tx * tx_float_count + t + s * 64 * tx_float_count);
                    this_wave_f32s[0] += other_wave_f32s[0];
                    this_wave_f32s[1] += other_wave_f32s[1];
                }
                *(vec2_fp32*)(lds + tx * tx_float_count + t) = this_wave_f32s;
            } else if constexpr (kHeadDim == 64) {
                float this_wave_f32s = *(float*)(lds + tx * tx_float_count + t);
                #pragma unroll
                for (int s = 1; s < (SPLIT_COUNT >> 6); ++s) {  // 0 wave accumulate data from other waves
                    float other_wave_f32s = *(float*)(lds + tx * tx_float_count + t + s * 64 * tx_float_count);
                    this_wave_f32s += other_wave_f32s;
                }
                *(float*)(lds + tx * tx_float_count + t) = this_wave_f32s;
            }
        }
        __syncthreads(); // here, __sync may not be necessary
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            if constexpr (kHeadDim % 128 == 0) {
                tx_accum[t]     = lds[tx * tx_float_count + t];
                tx_accum[t + 1] = lds[tx * tx_float_count + t + 1];
                vec2_Element<reduceType> accum_result;
                #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                    accum_result = DownCastPairNoPack<float, reduceType>(tx_accum[t], tx_accum[t + 1]);
                #else
                    accum_result[0] = DownCast<float, reduceType, true>(tx_accum[t]);
                    accum_result[1] = DownCast<float, reduceType, true>(tx_accum[t + 1]); // here, v_cvt_pkrtz can be used
                #endif
                *(vec2_Element<reduceType>*)(output_ptr + t) = accum_result;
            } else if constexpr (kHeadDim == 64) {
                tx_accum[t]  = lds[tx * tx_float_count + t];
                reduceType accum_result = DownCast<float, reduceType, false>(tx_accum[t]);
                *(reduceType*)(output_ptr + t) = accum_result;
            }
        }
    }
}


template<typename accumType, typename reduceType, const int SPLIT_COUNT, const bool UnRoll, const bool Tail, const int kHeadDim, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_reduce_varlen_kernel(
        Params params) {
    static_assert(SPLIT_COUNT <= 64 and (kHeadDim % 128 == 0 or kHeadDim == 64));
    constexpr int num_splits = SPLIT_COUNT;

    /*  bottleneck
            1. s_load_dword latency for args
            2. s_load_dword for cu_seqlens_q/cu_seqlens_k, cache miss
        solution:
            1. why kernel args cannot be hit on SQC data cache ? kernel packet different ?
            2. llvm-backend rescheduling, overlap args loading and cu_seqlens loading with better granularity
            3. asm
    */

    // 128 threads, each thread won't process more than 4 half data, 2 is appropriate, for 64 threads to processing 128 half
    __shared__ float lds[512];
    int tx               = threadIdx.x;
    int total_q          = params.total_q;
    int bidh_ngroup      = blockIdx.x;
    int total_h_ngroup   = gridDim.x;
    int s_m_split_stride = total_h_ngroup * total_q; // offset from the next split
    int bidh             = bidh_ngroup / params.ngroups;
    int group_id         = bidh_ngroup - bidh * params.ngroups;

    // recompute the true actual_seqlen_k and num_split
    const int bidb = blockIdx.y;
    int actual_seqlen_k = params.cu_seqlens_k[bidb];

    // varlen q
    int sum_s_q = params.cu_seqlens_q[bidb];
    int actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - sum_s_q;

    // compute partition_size when fix num_splits
    int partition_size = splitkv_get_partitionsize_of_fix_numsplits(actual_seqlen_k, params.num_splits);
    const int true_num_splits = Tail ? max(1, floor_div(actual_seqlen_k, partition_size)): ceil_div(actual_seqlen_k, partition_size);
    // const int true_num_splits = num_splits;

    bool exceed_split = (tx >= true_num_splits); // process boundary

    float* softmax_lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr);
    float* softmax_lseaccum_ptr = reinterpret_cast<float*>(params.softmax_lseaccum_ptr);

    for (int cur_s_q = 0; cur_s_q < actual_seqlen_q; ++cur_s_q) {

        // h * ngroups * (bs)
        int block_x = bidh_ngroup * total_q + sum_s_q + cur_s_q;

        // load local lse value for each split
        float lse_local = softmax_lseaccum_ptr[block_x + min(tx, num_splits - 1) * s_m_split_stride];
        __builtin_amdgcn_sched_barrier(0);

        // finally, do rescale for each split and reduce the sum of them
        // each block(1waves) process (num_splits x head_dim) elements in total
        // for head_dim 128, each thread process 2 halfs for num_splits times
        constexpr int tx_float_count = kHeadDim >> 6;
        float tx_accum[tx_float_count] = {0.f};
        // offset from the next split for output from previous kernel, split * (batch, head,seq) * headdim
        int oaccum_stride = s_m_split_stride * kHeadDim;
        // {total_q, ngroups, num_heads, -1} --> {total_q, num_heads, ngroups, -1}
        // int real_block_x  = (sum_s_q + cur_s_q) * total_h_ngroup + group_id * params.h + bidh;
        int real_block_x = (sum_s_q + cur_s_q) * total_h_ngroup + bidh * params.ngroups + group_id;
        int tx_offset = real_block_x * kHeadDim + (tx & 63) * tx_float_count;
        reduceType* output_ptr = reinterpret_cast<reduceType*>(params.o_ptr) + tx_offset;
        accumType* oaccum_ptr  = reinterpret_cast<accumType*>(params.oaccum_ptr);
        // prefetch all vgprs
        constexpr int tx_float_loop = tx_float_count >> 1;
        vec2_Element<accumType> load_vec[num_splits][tx_float_loop];
        accumType load[num_splits][tx_float_loop];
        // num_splits may not be 64, and thus need boundary judgement
        #pragma unroll
        for (int i = 0; i < num_splits; ++i) {
            #pragma unroll
            for (int t = 0; t < tx_float_count; t += 2) {
                if constexpr (kHeadDim % 128 == 0) {
                    load_vec[i][t >> 1] = *(vec2_Element<accumType>*)(oaccum_ptr + tx_offset + t);
                } else if constexpr (kHeadDim == 64) {
                    load[i][t >> 1]  = *(accumType*)(oaccum_ptr + tx_offset + t);
                }
            }
            // switch to next split
            tx_offset += oaccum_stride;
        }
        __builtin_amdgcn_sched_barrier(0);

        // process initialization as -inf
        if (exceed_split) {
            lse_local = -INFINITY;
        }
        // reduce max lse
        float lse_max_local = lse_local;
        #pragma unroll
        for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
            lse_max_local = max(lse_max_local, __shfl_xor_tmp(lse_max_local, step));
        }
        // reduce sum lse
        float lse_local_logsum = __expf(lse_local - lse_max_local);
        #pragma unroll
        for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
            lse_local_logsum = lse_local_logsum + __shfl_xor_tmp(lse_local_logsum, step);
        }
        lse_local_logsum = __logf(lse_local_logsum) + lse_max_local;

        // store softmax_lse
        if (tx == 0) {
            softmax_lse_ptr[block_x] = lse_local_logsum;
        }

        // store rescale coefficient into lds
        lds[tx] = __expf(lse_local - lse_local_logsum);

        // num_splits may not be 64, and thus need boundary judgement
        #pragma unroll
        for (int i = 0; i < num_splits; ++i) {
            // read ultimate scale value for current split
            float s_scale      = lds[i];
            bool within_splits = (i < true_num_splits);
            #pragma unroll
            for (int t = 0; t < tx_float_count; t += 2) {
                if constexpr (kHeadDim % 128 == 0) {
                    // half -> float32, reduce precision loss
                    float a_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load_vec[i][t >> 1][0]): 0.f;
                    float b_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load_vec[i][t >> 1][1]): 0.f;
                    // do rescale and sum
                    tx_accum[t]     = __llvm_fma_f32(a_f32, s_scale, tx_accum[t]);
                    tx_accum[t + 1] = __llvm_fma_f32(b_f32, s_scale, tx_accum[t + 1]);
                } else if constexpr (kHeadDim == 64) {
                    // half -> float32, reduce precision loss
                    float load_f32  = within_splits? splitkv_upcast_to_f32<accumType>(load[i][t >> 1]): 0.f;
                    // do rescale and sum
                    tx_accum[t]     = __llvm_fma_f32(load_f32, s_scale, tx_accum[t]);
                }
            }
        }
        // write results
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            if constexpr (kHeadDim % 128 == 0) {
                vec2_Element<reduceType> accum_result;
                #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                    accum_result = DownCastPairNoPack<float, reduceType>(tx_accum[t], tx_accum[t + 1]);
                #else
                    accum_result[0] = DownCast<float, reduceType, true>(tx_accum[t]);
                    accum_result[1] = DownCast<float, reduceType, true>(tx_accum[t + 1]); // here, v_cvt_pkrtz can be used
                #endif
                *(vec2_Element<reduceType>*)(output_ptr + t) = accum_result;
            } else if constexpr (kHeadDim == 64) {
                reduceType accum_result = DownCast<float, reduceType, false>(tx_accum[t]);
                output_ptr[t] = accum_result;
            }
        }
    }
}



template<typename accumType, typename reduceType, const int SPLIT_COUNT, const bool UnRoll, const bool Tail, const int kHeadDim, typename Params>
__global__ void __launch_bounds__(256, 1) flash_mla_splitkv_reduce_kernel(
        Params params) {
    static_assert(SPLIT_COUNT <= 64 and (kHeadDim % 128 == 0 or kHeadDim == 64));
    constexpr int WARP_NUM = 4;
    constexpr int lds_required_per_wave = (kHeadDim >> 2);
    constexpr int num_splits = SPLIT_COUNT; // UnRoll ? SPLIT_COUNT: params.num_splits;

    // acquire scalar data by sqc cache
    float* softmax_lse_ptr = params.softmax_lse_ptr;
    accumType* oaccum_ptr  = reinterpret_cast<accumType*>(params.oaccum_ptr);
    void* o_ptr            = params.o_ptr;
    int32_t* cu_seqlens_k  = params.cu_seqlens_k;
    int params_num_splits  = params.num_splits;
    int params_partition_size = params.partition_size;
    int h                  = params.h;
    int seqlen_q           = params.seqlen_q;
    int layout             = params.layout;

    // acquire task
    int tx               = threadIdx.x & 63;
    int wave_id          = threadIdx.x >> 6;
    int block_x          = blockIdx.x;
    int s_m_split_stride = gridDim.x; // offset from the next split
    const int bidb       = block_x / (h * seqlen_q);

    // prefetch buffer to overlap with s_load_dword*
    float lse_local      = softmax_lse_ptr[block_x + min(tx, num_splits - 1) * s_m_split_stride];

    // share rescale across threads
    __shared__ float lds_space[512 + 1024];
    float* lds = lds_space + wave_id * lds_required_per_wave;

    // recompute the true actual_seqlen_k and num_split
    int actual_seqlen_k = cu_seqlens_k[bidb];

    // for flashmla, 512 elements are engaged to 4 blocks
    // within each block, num_splits / WARM_NUM load transactions are engaged to each wave
    constexpr int tx_float_count = (kHeadDim >> 2) >> 6;
    float tx_accum[tx_float_count] = {0.f};
    // offset from the next split for output from previous kernel, split * (batch, head,seq) * headdim
    int oaccum_stride = s_m_split_stride * kHeadDim;
    // int tx_offset= block_x * kHeadDim + tx * tx_float_count;
    int in_batch_offset = block_x - bidb * h * seqlen_q;
    int bidh = in_batch_offset / seqlen_q;
    int bids = in_batch_offset - bidh * seqlen_q;
    int real_block_x = layout == 0 ? block_x/*bhsd layout*/: bidb * seqlen_q * h + bids * h + bidh/*bshd layout*/;
    int tx_offset = real_block_x * kHeadDim + tx * tx_float_count + blockIdx.y * (kHeadDim >> 2) + min(wave_id, num_splits - 1) * oaccum_stride;
    reduceType* output_ptr = reinterpret_cast<reduceType*>(o_ptr) + tx_offset;
    // fetch all data into vgprs
    constexpr int SPLITS_PER_WAVE = std::max<int32_t>(1, num_splits >> 2);
    vec2_Element<accumType> load[SPLITS_PER_WAVE][tx_float_count >> 1];
    #pragma unroll
    for (int i = 0; i < num_splits; i += WARP_NUM) {
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            load[i >> 2][t >> 1] = *(vec2_Element<accumType>*)(oaccum_ptr + tx_offset + t);
        }
        // switch to next split
        tx_offset += WARP_NUM * oaccum_stride;
    }

    // compute partition_size when fix num_splits
    int partition_size = params_partition_size > MLA_MAX_SPLITS ? splitkv_get_partitionsize_of_fix_numsplits(actual_seqlen_k, params.num_splits): params_partition_size;
    const int true_num_splits = Tail ? max(1, floor_div(actual_seqlen_k, partition_size)): ceil_div(actual_seqlen_k, partition_size);
    // const int true_num_splits = num_splits;

    bool exceed_split = (tx >= true_num_splits); // process boundary

    // process initialization as -inf
    if (exceed_split) {
        lse_local = -INFINITY;
    }
    // reduce max lse
    float lse_max_local = lse_local;
    #pragma unroll
    for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
        lse_max_local = max(lse_max_local, __shfl_xor_tmp(lse_max_local, step));
    }
    // reduce sum lse
    float lse_local_logsum = __expf(lse_local - lse_max_local);
    #pragma unroll
    for (int step = SPLIT_COUNT >> 1; step > 0; step = (step >> 1)) {
        lse_local_logsum = lse_local_logsum + __shfl_xor_tmp(lse_local_logsum, step);
    }
    lse_local_logsum = __logf(lse_local_logsum) + lse_max_local;

    // store rescale coefficient into lds
    lds[tx] = __expf(lse_local - lse_local_logsum);

    // num_splits may not be 64, and thus need boundary judgement
    #pragma unroll
    for (int i = 0; i < num_splits; i += WARP_NUM) {
        // read ultimate scale value for current split
        bool within_splits = ((i + wave_id) < true_num_splits);
        float s_scale      = num_splits >= WARP_NUM ? lds[i + wave_id]: (within_splits ? lds[i + wave_id]: 0.f);
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            // half -> float32, reduce precision loss
            float a_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[i >> 2][t >> 1][0]): 0.f;
            float b_f32 = within_splits? splitkv_upcast_to_f32<accumType>(load[i >> 2][t >> 1][1]): 0.f;
            // do rescale and sum
            tx_accum[t]     = __llvm_fma_f32(a_f32, s_scale, tx_accum[t]);
            tx_accum[t + 1] = __llvm_fma_f32(b_f32, s_scale, tx_accum[t + 1]);
        }
    }
    // reduce across 4 waves
    float *reduce_lds = lds_space + 512;
    #pragma unroll
    for (int t = 0; t < tx_float_count; t += 2) {
        reduce_lds[wave_id * lds_required_per_wave + tx + t * 64]       = tx_accum[t];
        reduce_lds[wave_id * lds_required_per_wave + tx + (t + 1) * 64] = tx_accum[t + 1];
    }
    __syncthreads();
    if (wave_id == 0) {
        // write results
        #pragma unroll
        for (int t = 0; t < tx_float_count; t += 2) {
            // get data from other wave
            #pragma unroll
            for (int neighbor = 1; neighbor < WARP_NUM; ++neighbor) {
                tx_accum[t]     += reduce_lds[neighbor * lds_required_per_wave + tx + t * 64];
                tx_accum[t + 1] += reduce_lds[neighbor * lds_required_per_wave + tx + (t + 1) * 64];
            }
            // cvt
            vec2_Element<reduceType> accum_result;
            #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                accum_result = DownCastPairNoPack<float, reduceType>(tx_accum[t], tx_accum[t + 1]);
            #else
                accum_result[0] = DownCast<float, reduceType, true>(tx_accum[t]);
                accum_result[1] = DownCast<float, reduceType, true>(tx_accum[t + 1]);
            #endif
            // storation
            *(vec2_Element<reduceType>*)(output_ptr + t) = accum_result;
        }
    }
}