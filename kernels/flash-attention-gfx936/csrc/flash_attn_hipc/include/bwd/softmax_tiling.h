#pragma once

#include "numeric_types.h"
#include "utils.h"

using namespace flash;

template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;

    #pragma unroll
    for (int ni = 0; ni < (WARP_N / 32); ++ni) {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 8;
                if (col_idx >= max_seqlen_k) {
                    #pragma unroll
                    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                        #pragma unroll
                        for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_causal(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] =
                            (col_idx > col_idx_limit_right) ? -INFINITY : tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_local(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q,
                                        const int window_size_left, const int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_left = std::max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        if (col_idx >= max_seqlen_k || row_idx >= max_seqlen_q ||
                            col_idx > col_idx_limit_right ||
                            (HasWSLeft && col_idx < (col_idx_limit_left - 1))) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

//32*32的tile，结果矩阵根据奇偶分开设计
//mask_type == 0：无mask
//mask_type == 1: mask矩阵右上角
//mask_type == 2: mask矩阵左下角
template <bool Is_even_MN, int mask_type>
inline __device__ void apply_mask_bwd(union_vec4_fp32 tensor[1][4], int M, int N, int M_minus_N, int window_size_left, int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int lane_m_idx = (lane_id & 15);
    const int lane_n_idx = (lane_id >> 4);
    //无mask，仅进行边界判断
    if(!Is_even_MN && mask_type == 0) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = lane_n_idx * 2 + min_tile_n + vec_idx * 8;
                    if(N_offset > N - 1){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //mask右上角
    if (mask_type == 1 && (!Is_even_MN || Is_even_MN && std::abs(M_minus_N) < 128)) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = lane_m_idx * 2 + min_tile_m;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = lane_n_idx * 2 + min_tile_n + vec_idx * 8;
                    int N_limit = Is_even_MN ? (M_offset + M_minus_N) : min(N - 1, M_offset + M_minus_N);
                    if(N_offset > N_limit){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //mask左下角
    if (mask_type == 2 && (!Is_even_MN || Is_even_MN && std::abs(M_minus_N) < 128)) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = lane_m_idx * 2 + min_tile_m;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = lane_n_idx * 2 + min_tile_n + vec_idx * 8;
                    int N_limit = (M_offset + M_minus_N);
                    if((!Is_even_MN && N_offset > N - 1) || N_offset < N_limit){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //local mask
    if (mask_type == 3) {// && (!Is_even_MN || Is_even_MN && (std::abs(M_minus_N - window_size_left) < 128 || std::abs(M_minus_N + window_size_right) < 128))
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = lane_m_idx * 2 + min_tile_m;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = lane_n_idx * 2 + min_tile_n + vec_idx * 8;
                    int N_limit_left = (M_offset + M_minus_N - window_size_left);
                    int N_limit_right = (M_offset + M_minus_N + window_size_right);
                    if((!Is_even_MN && N_offset > N - 1) || N_offset < N_limit_left || N_offset > N_limit_right){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
}

//32*32的tile，结果矩阵根据mmac_4interleave设计
//mask_type == 0：无mask
//mask_type == 1: mask矩阵右上角
//mask_type == 2: mask矩阵左下角
template <bool Is_even_MN, int mask_type>
inline __device__ void apply_mask_bwd_gfx938(union_vec4_fp32 tensor[1][4], int M, int N, int M_minus_N, int window_size_left, int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int lane_m_idx = (lane_id & 15);
    const int lane_n_idx = (lane_id >> 4);
    //无mask，仅进行边界判断
    if(!Is_even_MN && mask_type == 0) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = min_tile_n * 16 + lane_n_idx * 4 + vec_idx;
                    if(N_offset > N - 1){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //mask右上角
    if (mask_type == 1 && (!Is_even_MN || Is_even_MN && std::abs(M_minus_N) < 128)) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = min_tile_m * 16 + lane_m_idx;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = min_tile_n * 16 + lane_n_idx * 4 + vec_idx;
                    int N_limit = Is_even_MN ? (M_offset + M_minus_N) : min(N - 1, M_offset + M_minus_N);
                    if(N_offset > N_limit){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //mask左下角
    if (mask_type == 2 && (!Is_even_MN || Is_even_MN && std::abs(M_minus_N) < 128)) {
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = min_tile_m * 16 + lane_m_idx;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = min_tile_n * 16 + lane_n_idx * 4 + vec_idx;
                    int N_limit = (M_offset + M_minus_N);
                    if((!Is_even_MN && N_offset > N - 1) || N_offset < N_limit){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
    //local mask
    if (mask_type == 3) {// && (!Is_even_MN || Is_even_MN && (std::abs(M_minus_N - window_size_left) < 128 || std::abs(M_minus_N + window_size_right) < 128))
        for(int min_tile_m = 0; min_tile_m < 2; min_tile_m ++) {
            int M_offset = min_tile_m * 16 + lane_m_idx;
            for(int min_tile_n = 0; min_tile_n < 2; min_tile_n ++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx ++) {
                    int N_offset = min_tile_n * 16 + lane_n_idx * 4 + vec_idx;
                    int N_limit_left = (M_offset + M_minus_N - window_size_left);
                    int N_limit_right = (M_offset + M_minus_N + window_size_right);
                    if((!Is_even_MN && N_offset > N - 1) || N_offset < N_limit_left || N_offset > N_limit_right){
                        tensor[0][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                    }
                }
            }
        }
    }
}

template <bool encode_dropout_in_sign_bit=false, typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_dropout(DataType tensor[(WARP_M/32)*(WARP_N/32)][4], uint8_t p_dropout_in_uint8_t,
                                     unsigned long long seed, unsigned long long offset,
                                     int block_col_start, int block_row_start,
                                     int block_col_stride) {
    // tensor has shape (8, MMA_M, MMA_N / 2)
    auto encode_dropout = [](bool keep, DataType val) {
        if (keep) { return val; }
        if constexpr (encode_dropout_in_sign_bit) {
            if constexpr (std::is_same<DataType, union_vec4_fp32>::value) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    val.f32[i] = flash::encode_dropout_sign_bit(val.f32[i]);
                }
                return val;
            } else {
                return -val;
            }
        } else {
            return DataType(0);
        }
    };
    // static_assert(decltype(size<2>(tensor))::value % 2 == 0);
    const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
    const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
    // if (cute::thread0()) { printf("threshold2 = 0x%x\n", p_dropout_8bit_in_uint32_t); }
    #pragma unroll
    for (int n = 0; n < (WARP_N/32); ++n, block_col_start += block_col_stride) {
        uint2 rowcol = make_uint2(block_row_start, block_col_start);
        #pragma unroll
        for (int m = 0; m < (WARP_M/32); ++m,  ++rowcol.x) {
            // if (cute::thread(32, 0)) { printf("m = %d, n = %d, row = %d, col = %d\n", m, n, int(rowcol.x), int(rowcol.y));}
            uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
            // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
            uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
            // Special implementation for 16-bit types: we duplicate the threshold to the
            // low and high 16 bits of a 32-bit value, then use the f16x2 comparison instruction
            // to get a mask. The low 16 bits of the mask will be either 0xffff or 0x0000,
            // and the high 16 bits will be either 0xffff or 0x0000, depending on whether
            // the random value is less than the threshold.
            // We then do a bit-wise AND between the mask and the original value (in 32-bit).
            // We're exploiting the fact that floating point comparison is equivalent to integer
            // comparison, since we're comparing unsigned integers whose top 8-bits are zero.
            if (!encode_dropout_in_sign_bit
                && (std::is_same<DataType, Float16>::value || std::is_same<DataType, BFloat16>::value)) {
                // uint16_t rnd_16[16];
                // #pragma unroll
                // for (int i = 0; i < 16; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                // uint32_t (&rnd_32)[8] = reinterpret_cast<uint32_t (&)[8]>(rnd_16);
                // #pragma unroll
                // for (int j = 0; j < 2; j++) {
                //     Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                //     // if (cute::thread0()) { printf("random = 0x%x, 0x%x, 0x%x, 0x%x\n", rnd_32[j * 4 + 0], rnd_32[j * 4 + 1], rnd_32[j * 4 + 2], rnd_32[j * 4 + 3]); }
                //     // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                //     #pragma unroll
                //     for (int i = 0; i < 4; i++) {
                //         uint32_t mask;
                //         asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[j * 4 + i]), "r"(p_dropout_8bit_in_uint32_t));
                //         tensor_uint32(i) &= mask;
                //     }
                //     // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                // }
            } else {
                //min tile for a warp is 32*32
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; n_idx++) {
                    #pragma unroll
                    for (int m_idx = 0; m_idx < 2; m_idx++) {
                        for(int vec_idx=0; vec_idx<4; vec_idx++) { //mmac min_tile is 16*16, a warp is 64 thread
                            tensor[(n*(WARP_N/16)*(WARP_M/16) + m*(WARP_M/16)) + n_idx * 2 + m_idx][vec_idx] = encode_dropout(rnd_8[n_idx * 8 + m_idx] <= p_dropout_in_uint8_t, tensor[(n*(WARP_N/16)*(WARP_M/16) + m*(WARP_M/16)) + n_idx * 2 + m_idx][vec_idx]);
                        }
                    }
                    // Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                    // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                }
            }
            // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            // //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
            // // }
        }
    }
}

template <bool Encode_dropout_in_sign_bit=false, typename DataType, int WARP_M, int WARP_N, int BLOCK_ROW_STRIDE, bool Is_even_MN>
inline __device__ void apply_dropout(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_q, const int max_seqlen_k,
                                     const int row_idx_offset_, const int col_idx_offset_, const int batch_head_idx,
                                     unsigned long long seed, unsigned long long offset, uint32_t p_dropout_in_8bits_value,
                                     union_vec2_uint rowcol, uint32_t* dropout_debug_count) {
    const int lane_id = threadIdx.x & 63;
    const int lane_row = lane_id & 15;
    const int lane_col_group = lane_id >> 4;
    const int row_idx_offset = row_idx_offset_ + lane_row * 2;
    const int col_idx_offset = col_idx_offset_ + lane_col_group * 2;
    const uint32_t p_dropout_threshold = min(255u, p_dropout_in_8bits_value + 1u);
    const unsigned long long offset_base = offset - static_cast<unsigned long long>(lane_id);
    (void)rowcol;
    (void)dropout_debug_count;
    (void)BLOCK_ROW_STRIDE;
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for (uint32_t ni = 0; ni < (WARP_N / 32); ++ni) {
            #pragma unroll
            for(uint32_t vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int abs_q_base = row_idx_offset + mi * 32;
                const int abs_k_base = col_idx_offset + ni * 32 + vec_idx * 8;
                const uint32_t rand_pack = flash::dropout_rand_2x2_u32(seed, offset_base, batch_head_idx, abs_q_base, abs_k_base);
                #pragma unroll
                for(uint32_t min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int abs_k = abs_k_base + min_tile_n;
                    #pragma unroll
                    for(uint32_t min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        const int abs_q = abs_q_base + min_tile_m;
	                        if constexpr (Is_even_MN) {
	                            uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
	                            if (cur_rand > p_dropout_threshold) {
	                                if constexpr (Encode_dropout_in_sign_bit) {
	                                    tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = flash::encode_dropout_sign_bit(tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
	                                } else {
	                                    tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = 0x0;
	                                }
	                            }
	                        } else if constexpr (not Is_even_MN) {
	                            if (abs_q < max_seqlen_q && abs_k < max_seqlen_k) {
	                                uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
	                                if (cur_rand > p_dropout_threshold) {
	                                    if constexpr (Encode_dropout_in_sign_bit) {
	                                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = flash::encode_dropout_sign_bit(tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
	                                    } else {
	                                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = 0x0;
	                                    }
	                                }
	                            }
	                        }
                    }
                }
            }
        }
    }
}

template <bool Encode_dropout_in_sign_bit=false, typename DataType, int WARP_M, int WARP_N, int BLOCK_ROW_STRIDE, bool Is_even_MN>
inline __device__ void apply_dropout_transposed_kq(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_q, const int max_seqlen_k,
                                                   const int row_idx_offset_, const int col_idx_offset_, const int batch_head_idx,
                                                   unsigned long long seed, unsigned long long offset, uint32_t p_dropout_in_8bits_value,
                                                   union_vec2_uint rowcol, uint32_t* dropout_debug_count) {
    const int lane_id = threadIdx.x & 63;
    const int lane_low = lane_id & 15;
    const int lane_high = lane_id >> 4;
    const uint32_t p_dropout_threshold = min(255u, p_dropout_in_8bits_value + 1u);
    const unsigned long long offset_base = offset - static_cast<unsigned long long>(lane_id);
    (void)rowcol;
    (void)dropout_debug_count;
    (void)BLOCK_ROW_STRIDE;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for (uint32_t ni = 0; ni < (WARP_N / 32); ++ni) {
            #pragma unroll
            for (uint32_t vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int abs_q_base = row_idx_offset_ + mi * 32 + vec_idx * 8 + lane_high * 2;
                const int abs_k_base = col_idx_offset_ + ni * 32 + (lane_low << 1);
                const uint32_t rand_pack = flash::dropout_rand_2x2_u32(seed, offset_base, batch_head_idx, abs_q_base, abs_k_base);
                #pragma unroll
                for (uint32_t min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    const int abs_q = abs_q_base + min_tile_m;
                    #pragma unroll
                    for (uint32_t min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        const int abs_k = abs_k_base + min_tile_n;
	                        if constexpr (Is_even_MN) {
	                            uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
	                            if (cur_rand > p_dropout_threshold) {
	                                if constexpr (Encode_dropout_in_sign_bit) {
	                                    tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx] = flash::encode_dropout_sign_bit(tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx]);
	                                } else {
	                                    tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx] = 0.0f;
	                                }
	                            }
	                        } else {
	                            if (abs_q < max_seqlen_q && abs_k < max_seqlen_k) {
	                                uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
	                                if (cur_rand > p_dropout_threshold) {
	                                    if constexpr (Encode_dropout_in_sign_bit) {
	                                        tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx] = flash::encode_dropout_sign_bit(tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx]);
	                                    } else {
	                                        tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx] = 0.0f;
	                                    }
	                                }
	                            } else if constexpr (!Encode_dropout_in_sign_bit) {
	                                tensor[ni + mi * (WARP_N / 32)][min_tile_m * 2 + min_tile_n].f32[vec_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int BLOCK_M, int WARP_N, typename ElementType, bool Is_even_MN>
inline __device__ void convert_pk_type_dropout_dv(union_vec2_f16x2<ElementType> p_reg[(BLOCK_M / 32) * (WARP_N / 32)][4],
                                                  union_vec4_fp32 s_reg[(BLOCK_M / 32) * (WARP_N / 32)][4],
                                                  const int max_seqlen_q, const int max_seqlen_k,
                                                  const int row_idx_offset_, const int col_idx_offset_,
                                                  const int batch_head_idx, unsigned long long seed,
                                                  unsigned long long offset, uint32_t p_dropout_in_8bits_value,
                                                  float rp_dropout) {
    const int lane_id = threadIdx.x & 63;
    const int lane_row = lane_id >> 4;
    const int lane_col = lane_id & 15;
    const int row_idx_offset = row_idx_offset_ + lane_row * 2;
    const int col_idx_offset = col_idx_offset_ + lane_col * 2;
    const uint32_t p_dropout_threshold = min(255u, p_dropout_in_8bits_value + 1u);
    const unsigned long long offset_base = offset - static_cast<unsigned long long>(lane_id);

    #pragma unroll
    for (int m_idx = 0; m_idx < (BLOCK_M / 32); ++m_idx) {
        #pragma unroll
        for (int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
            const int reg_idx = n_idx + m_idx * (WARP_N / 32);
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int abs_q_base = row_idx_offset + m_idx * 32 + vec_idx * 8;
                const int abs_k_base = col_idx_offset + n_idx * 32;
                const uint32_t rand_pack = flash::dropout_rand_2x2_u32(seed, offset_base, batch_head_idx, abs_q_base, abs_k_base);
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        const int abs_q = abs_q_base + min_tile_n;
                        const int abs_k = abs_k_base + min_tile_m;
                        const int value_idx = min_tile_n * 2 + min_tile_m;
                        bool drop = true;
                        if constexpr (Is_even_MN) {
                            drop = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k)) > p_dropout_threshold;
                        } else {
                            if (abs_q < max_seqlen_q && abs_k < max_seqlen_k) {
                                drop = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k)) > p_dropout_threshold;
                            }
	                        }
	                        const float p = clear_dropout_encoded_sign(s_reg[reg_idx][value_idx].f32[vec_idx]);
	                        p_reg[reg_idx][value_idx].f16x2[vec_idx >> 1][vec_idx & 1] =
	                            DownCast<float, ElementType, true>(drop ? 0.0f : p * rp_dropout);
	                    }
	                }
	            }
	        }
    }
}


template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void thread_reduce_(const DataType0 tensor[(WARP_M/32)*(WARP_N/32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                summary[m_idx*2 + min_tile_m] = (OpType==0)? 0 : -INFINITY;  //OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        for(int vec_idx=0; vec_idx<4; vec_idx++) { //mmac min_tile is 16*16, a warp is 64 thread
                            summary[m_idx*2 + min_tile_m] = op(summary[m_idx*2 + min_tile_m], tensor[m_idx + n_idx*(WARP_M/32)][min_tile_n*2 + min_tile_m][vec_idx]);
                        }
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                summary_cur[m_idx*2 + min_tile_m] = summary[m_idx*2 + min_tile_m];// op(summary[m_idx*2 + min_tile_m], tensor[m_idx][min_tile_m][0]);
                #pragma unroll
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        for(int vec_idx=0; vec_idx<4; vec_idx++) { //mmac min_tile is 16*16, a warp is 64 thread
                            summary_cur[m_idx*2 + min_tile_m] = op(summary_cur[m_idx*2 + min_tile_m], tensor[m_idx + n_idx*(WARP_M/32)][min_tile_n*2 + min_tile_m][vec_idx]);
                        }
                    }
                }
            }
        }
    }
}

template<typename Operator, typename DataType, int WARP_M>
__device__ inline void quad_allreduce_(DataType *dst, DataType *src, Operator &op) {
    #pragma unroll
    for (int i = 0; i < (WARP_M/16); i++) {
        dst[i] = Allreduce<64>::run(src[i], op);
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_(const DataType0 tensor[(WARP_M/32)*(WARP_N/32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        thread_reduce_<true, Operator, OpType, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op);
        quad_allreduce_<Operator, DataType1, WARP_M>(summary, summary, op);
    } else {
        thread_reduce_<false, Operator, OpType, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op, summary_cur);
        quad_allreduce_<Operator, DataType1, WARP_M>(summary_cur, summary_cur, op);
    }
}

//zero_init==true, max is current max_score, max_cur=nullptr
//zero_init==true, max is prev max_score, max_cur!=nullptr
template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_max(const DataType0 tensor[(WARP_M/32)*(WARP_N/32)][4], DataType1 *max , DataType1 *max_cur=nullptr) {
    MaxOp<float> max_op;
    if(zero_init == true) {
        reduce_<true, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, max, max_op);
    } else {
        reduce_<false, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, max, max_op, max_cur);
    }
}

template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_sum(DataType0 tensor[(WARP_M/32)*(WARP_N/32)][4], DataType1 *sum,  DataType1 *sum_cur=nullptr){
    SumOp<float> sum_op;
    if(zero_init == true) {
        reduce_<true, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, sum, sum_op);
    } else {
        reduce_<false, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, sum, sum_op, sum_cur);
    }
}


// Apply the exp to all the elements.
template <bool Scale_max=true, int BLOCK_M, int WARP_N, typename DataType0, typename DataType1>
inline __device__ void scale_apply_exp2_bwd(DataType0 tensor[(BLOCK_M/32)*(WARP_N/32)][4], const DataType1 *max, const float scale) {
    // #if defined(__gfx936__)
    //     auto vec2_scale = vec2_fp32{scale, scale};
    // #endif

    #pragma unroll
    for (int mi = 0; mi < (BLOCK_M/32); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
            for(int vec_idx=0; vec_idx<4; vec_idx++) {
                const float max_scaled = (max[(mi*2 + min_tile_m)*4 + vec_idx] * (Scale_max ? scale : float(M_LOG2E)));
                // #if defined(__gfx936__)
                //     auto vec2_max_scaled = vec2_fp32{-max_scaled, -max_scaled};
                // #endif
                #pragma unroll
                for (int ni = 0; ni < (WARP_N/32); ++ni)  {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                    //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                    #if 0//defined(__gfx936__)
                        auto vec2_tensor = vec2_fp32{tensor[ni + mi*(WARP_N/32)][min_tile_m*2].f32[vec_idx], tensor[ni + mi*(WARP_N/32)][min_tile_m*2 + 1].f32[vec_idx]};
                        auto vec2_scale = vec2_fp32{scale, scale};
                        auto vec2_max_scaled = vec2_fp32{-max_scaled, -max_scaled};
                        auto tensor_tmp =
                            __builtin_hcu_pk_fma_f32(
                                vec2_tensor,
                                vec2_scale,
                                vec2_max_scaled);
                            // __builtin_hcu_v_pk_fma_f32(
                            //     vec2_tensor,
                            //     vec2_scale,
                            //     vec2_max_scaled);
                        tensor[ni + mi*(WARP_N/32)][min_tile_m*2].f32[vec_idx] = __llvm_exp2_f32(tensor_tmp[0]);
                        tensor[ni + mi*(WARP_N/32)][min_tile_m*2 + 1].f32[vec_idx] = __llvm_exp2_f32(tensor_tmp[1]);
                    #else
                        #pragma unroll
                        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                      //使用__llvm_exp2_f32会产生nan，使用exp2f则没问题
                            // tensor[ni + mi*(WARP_N/32)][min_tile_n + min_tile_m*2].f32[vec_idx] =exp2f(tensor[ni + mi*(WARP_N/32)][min_tile_n + min_tile_m*2].f32[vec_idx] * scale - max_scaled);
                            tensor[ni + mi*(WARP_N/32)][min_tile_n + min_tile_m*2].f32[vec_idx] =__llvm_exp2_f32(tensor[ni + mi*(WARP_N/32)][min_tile_n + min_tile_m*2].f32[vec_idx] * scale - max_scaled);
                        }
                    #endif
                }
            }
        }
    }
}


// Apply the exp to all the elements.
template <bool Scale_max=true, int WARP_M, int BLOCK_N, typename DataType0, typename DataType1>
inline __device__ void scale_apply_exp2_bwd_seq_q_major(DataType0 tensor[(BLOCK_N/32)*(WARP_M/32)][4], const DataType1 max[WARP_M/16], const float scale) {
    // const float max_scaled = max[0] * float(M_LOG2E);
    #pragma unroll
    for (int ni = 0; ni < (BLOCK_N/32); ++ni)  {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for (int mi = 0; mi < (WARP_M/32); ++mi) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {

                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    const float max_scaled = (max[mi*2 + min_tile_m] * (Scale_max ? scale : float(M_LOG2E)));

                    #pragma unroll
                    for(int vec_idx=0; vec_idx<4; vec_idx++) {
                        tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] =
                        __llvm_exp2_f32(tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] * scale - max_scaled);
                        // tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] =
                        // exp2f(tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] * scale - max_scaled);
                        // tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] =
                        // __llvm_exp2_f32(tensor[mi + ni*(WARP_M/32)][min_tile_n*2 + min_tile_m].f32[vec_idx] * scale - max_scaled + 64) * __llvm_exp2_f32(-64);
                    }
                }
            }
        }
    }
}



#if 0
template<bool Is_first, bool Check_inf=false, typename DataType0, typename DataType1,int K/*head_dim*/, int kBlockK, int WARP_M, int WARP_N>
inline __device__ void softmax_rescale_o(DataType0 scores[(WARP_N/32)*(WARP_M/32)][4], DataType1 *scores_max, DataType1 *scores_sum,
                                         DataType0 acc_o[(K/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4], float softmax_scale_log2) {
    if (Is_first) {
        reduce_max</*zero_init=*/true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max);
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max, softmax_scale_log2);
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_sum);
    } else {
        float scores_max_cur[WARP_M/16]; //calculate max of each row
        reduce_max</*zero_init=*/false, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max, scores_max_cur); // scores_max is prev scores max

        for (int mi = 0; mi < (WARP_M/32); ++mi) {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // If we don't have float around M_LOG2E the multiplication is done in fp64.
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                float scores_max_cur_reg = !Check_inf
                        ? scores_max_cur[mi*2 + min_tile_m]
                        : (scores_max_cur[mi*2 + min_tile_m] == -INFINITY ? 0.0f : scores_max_cur[mi*2 + min_tile_m]);

                float scores_scale = __llvm_exp2_f32((scores_max[mi*2 + min_tile_m] - scores_max_cur_reg) * softmax_scale_log2);
                scores_sum[mi*2 + min_tile_m] *= scores_scale;

                #pragma unroll
                for(int pv_n_loop=0; pv_n_loop<(K/kBlockK); pv_n_loop++)  {
                    #pragma unroll
                    for (int ni = 0; ni < (kBlockK/32); ++ni)  {
                            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                            // max * log_2(e)) This allows the compiler to use the ffma
                            // instruction instead of fadd and fmul separately.
                            for(int vec_idx=0; vec_idx<4; vec_idx++) {
                                //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                                #pragma unroll
                                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                                    acc_o[pv_n_loop * ((WARP_M/32)*(kBlockK/32)) + (mi + ni*(WARP_M/32))][min_tile_n*2 + min_tile_m][vec_idx] = acc_o[pv_n_loop * ((WARP_M/32)*(kBlockK/32)) + (mi + ni*(WARP_M/32))][min_tile_n*2 + min_tile_m][vec_idx] * scores_scale;
                                }
                            }
                    }
                }
            }
        }
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max_cur, softmax_scale_log2);

        float scores_sum_cur[WARP_M/16]={0.0f};
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_sum_cur);

        #pragma unroll
        for (int mi = 0; mi < (WARP_M/16); ++mi) { scores_sum[mi] += scores_sum_cur[mi]; }


    }
};
#endif
