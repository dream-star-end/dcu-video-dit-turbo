// Copyright (c) 2023, Tri Dao.

#pragma once

// #include <ATen/cuda/CUDAContext.h>
#include <config.h>
#include <static_switch.h>
#include <flash.h>
#include <numeric_types.h>
#include <block_info.h>
#include <kernel_traits.h>
#include <bwd/flash_attention_bwd.h>

template<bool Clear_dQaccum=true, bool Is_even_MN, class Element, class ElementAccumType,  int kBlockM_, int kBlockN_, int WARP_M_, int WARP_N_, int kHeadDim_, int STAGES_, bool USE_BSHD_LAYOUT, typename Params>
__global__ void  __launch_bounds__(256,1) flash_bwd_dot_do_o_kernel(Params params) {
    #if defined(__gfx946__)
    compute_dot_do_o_gfx946<true, Is_even_MN, Element, ElementAccumType, kBlockM_, kBlockN_, WARP_M_, WARP_N_, kHeadDim_, STAGES_, USE_BSHD_LAYOUT>(params);
    #elif defined(__gfx938__)
    compute_dot_do_o_gfx938<true, Is_even_MN, Element, ElementAccumType, kBlockM_, kBlockN_, WARP_M_, WARP_N_, kHeadDim_, STAGES_, USE_BSHD_LAYOUT>(params);
    #else
    compute_dot_do_o<true, Is_even_MN, Element, ElementAccumType, kBlockM_, kBlockN_, WARP_M_, WARP_N_, kHeadDim_, STAGES_, USE_BSHD_LAYOUT>(params);
    #endif
}

template<class Element, class ElementAccumType, bool Is_dropout, bool Is_causal , bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel,  int kBlockM_, int kBlockN_, int K, int K_v, int kBlockK_, int WARP_M_, int WARP_N_, bool USE_BSHD_LAYOUT, typename Params>
__global__ void  __launch_bounds__(256,1) flash_attention_dv_dk_bwd_kernel(Params params
    ) {
    const int bidbh = blockIdx.x + blockIdx.z * params.se_balance_cnt;
    const int bidb = bidbh / params.h;
    const int bidh = bidbh % params.h;
    const int n_block = blockIdx.y;
    #if defined(__gfx946__)
    compute_dk_dv_1colblock_gfx946<Element, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, USE_BSHD_LAYOUT>(params, bidb, bidh, n_block);
    #elif defined(__gfx938__)
    if constexpr (!Is_dropout) {
        if (params.cu_seqlens_q == nullptr && K == 128 && Is_even_K) {
            compute_dk_dv_1colblock_gfx938<Element, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, USE_BSHD_LAYOUT>(params, bidb, bidh, n_block);
        } else {
            compute_dk_dv_1colblock<Element, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, USE_BSHD_LAYOUT>(params, bidb, bidh, n_block);
        }
    } else {
        compute_dk_dv_1colblock<Element, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, USE_BSHD_LAYOUT>(params, bidb, bidh, n_block);
    }
    #else
    compute_dk_dv_1colblock<Element, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, USE_BSHD_LAYOUT>(params, bidb, bidh, n_block);
    #endif
}

template<class ElementType, class ElementAccumType, bool Is_dropout, bool Is_causal , bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel,  int kBlockM_, int kBlockN_, int K, int K_v, int kBlockK_, int WARP_M_, int WARP_N_, int STAGES, int USE_BSHD_LAYOUT, typename Params>
__global__ void  __launch_bounds__(256,1) flash_attention_dq_bwd_kernel(Params params) {
    const int bidbh = blockIdx.x + blockIdx.z * params.se_balance_cnt;
    const int bidb = bidbh / params.h;
    const int bidh = bidbh % params.h;

    const int m_actual_block = (params.seqlen_q + kBlockM_ - 1) / kBlockM_;
    const int m_block = m_actual_block - 1 - blockIdx.y;
    #if defined(__gfx946__)
    compute_dq_1colblock_gfx946<ElementType, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, STAGES, USE_BSHD_LAYOUT>(params, bidb, bidh, m_block);
    #elif defined(__gfx938__)
    if constexpr (!Is_dropout) {
        if (params.cu_seqlens_q == nullptr) {
            compute_dq_1colblock_gfx938<ElementType, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, STAGES, USE_BSHD_LAYOUT>(params, bidb, bidh, m_block);
        } else {
            compute_dq_1colblock<ElementType, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, STAGES, USE_BSHD_LAYOUT>(params, bidb, bidh, m_block);
        }
    } else {
        compute_dq_1colblock<ElementType, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, STAGES, USE_BSHD_LAYOUT>(params, bidb, bidh, m_block);
    }
    #else
    compute_dq_1colblock<ElementType, float, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, K_v, kBlockK_, WARP_M_, WARP_N_, STAGES, USE_BSHD_LAYOUT>(params, bidb, bidh, m_block);
    #endif
}

template<bool Is_causal , bool Is_local, bool Is_dropout, typename Kernel_traits, bool Is_first, bool Is_last, bool Seq_parallel=false, bool USE_BSHD_LAYOUT, typename Flash_bwd_params>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, hipStream_t stream) {
    using Element          = typename Kernel_traits::Element;
    using ElementAccumType = typename Kernel_traits::ElementAccum;
    constexpr int K        = Kernel_traits::kHeadDim;
    constexpr int K_v      = Kernel_traits::kHeadDimV;
    constexpr int kBlockM_ = Kernel_traits::kBlockM;
    constexpr int kBlockN_ = Kernel_traits::kBlockN;
    constexpr int kBlockK_ = Kernel_traits::kBlockK;
    constexpr int WARP_M_  = Kernel_traits::kWaveM;
    constexpr int WARP_N_  = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;

    hipModule_t hModule = NULL;
    hipFunction_t hFunction;
    hipError_t ret = hipSuccess;

    // 获取cu数量
    auto& instance = DeviceProperties<Kernel_traits, FAFUNC::BACKWARD>::GetInstance();
    size_t sharedMemSize = instance.lds_size;
    params.cu_count      = instance.cu_count;
    if constexpr (K <= 128) {
        if (instance.gcn_arch == 938 && params.cu_seqlens_q == nullptr) {
            sharedMemSize = (K == 128 && !Is_dropout) ? 16 * 1024 : 32 * 1024;
        }
    }
    // std::cout << "sharedMemSize: " << (sharedMemSize/1024) << "KB" << std::endl;
    // std::cout << "cu_count: " << cu_count << std::endl;
    // std::cout<<"do_row_stride = "<<params.do_row_stride<<" do_head_stride = "<<params.do_head_stride<<std::endl;
    // std::cout<<"o_row_stride = "<<params.o_row_stride<<" o_head_stride = "<<params.o_head_stride<<std::endl;
    // std::cout<<"q_row_stride = "<<params.q_row_stride<<" q_head_stride = "<<params.q_head_stride<<std::endl;
    // std::cout<<"k_row_stride = "<<params.k_row_stride<<" k_head_stride = "<<params.k_head_stride<<std::endl;
    // std::cout<<"v_row_stride = "<<params.v_row_stride<<" v_head_stride = "<<params.v_head_stride<<std::endl;
    // std::cout<<"USE_BSHD_LAYOUT="<<USE_BSHD_LAYOUT<<std::endl;
    const bool is_even_MN = ((params.seqlen_k % kBlockN_) == 0) && ((params.seqlen_q) % kBlockM_ == 0) && params.cu_seqlens_q == nullptr;
    //is_even_K指headdim是否是32的整数倍，否则需要进行边界判断
    const bool is_even_K = params.d == K;

    const int num_m_block = (params.seqlen_q + kBlockM_ - 1) / kBlockM_;
    dim3 grid_m(num_m_block, params.h, params.b);
    int kMThreads = (kBlockM_+WARP_M_-1)/WARP_M_ * 64;

#if defined(FA_KERNEL_TIMER)
    hipEvent_t start, stop;
    check_hip_error(hipEventCreate( &start ));
    check_hip_error(hipEventCreate( &stop ));
    check_hip_error(hipEventRecord( start, 0));
#endif
    // BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    //     BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
    //             flash_bwd_dot_do_o_kernel<true, IsEvenMNConst, Element, ElementAccumType, kBlockM_, kBlockN_, WARP_M_, WARP_N_, K, STAGES>
    //                 <<<grid_m, kMThreads, 0, stream>>>(params);
    //             flash_attention_bwd<Element, float, Is_dropout, Is_causal, Is_local, IsEvenMNConst, IsEvenKConst, Is_first, Is_last, Seq_parallel,  kBlockM_, kBlockN_, K, kBlockK_, WARP_M_, WARP_N_, STAGES>
    //                 <<<grid_n, dimBlock, sharedMemSize, stream>>>(params);
    //             flash_bwd_convert_dq_kernel<IsEvenMNConst, Element, float, kBlockM_, kBlockN_, K, kBlockK_, WARP_M_, WARP_N_, STAGES>
    //                 <<<grid_m, kMThreads, sharedMemSize, stream>>>(params);
    //     });
    // });
    // total: 35.6 ms
    // flash_bwd_dot_do_o_kernel:   0.7 ms
    // flash_attention_bwd:        34.9 ms
    // flash_bwd_convert_dq_kernel: 0.9 ms

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        flash_bwd_dot_do_o_kernel<true,IsEvenMNConst, Element, ElementAccumType, kBlockM_, kBlockN_, WARP_M_, WARP_N_, K_v, STAGES, USE_BSHD_LAYOUT>
            <<<grid_m, kMThreads, 0, stream>>>(params);
    });

    int b_h_num = params.b * params.h;
    // if(b_h_num % 32 == 0){
    //     params.se_balance_cnt = 32;
    // } else if(b_h_num % 16 == 0){
    //     params.se_balance_cnt = 16;
    // } else
    if(b_h_num % 13 == 0){
        params.se_balance_cnt = 13;
    } else if(b_h_num % 9 == 0){
        params.se_balance_cnt = 9;
    } else if(b_h_num % 8 == 0){
        params.se_balance_cnt = 8;
    } else if(b_h_num % 7 ==0){
        params.se_balance_cnt = 7;
    } else if(b_h_num % 6 ==0){
        params.se_balance_cnt = 6;
    } else if(b_h_num % 5 ==0){
        params.se_balance_cnt = 5;
    } else if(b_h_num % 4 ==0){
        params.se_balance_cnt = 4;
    } else if(b_h_num % 3 ==0){
        params.se_balance_cnt = 3;
    } else if(b_h_num % 2 ==0){
        params.se_balance_cnt = 2;
    } else {
        params.se_balance_cnt = 1;
    }

    {
        constexpr int dk_dv_kBlockM = 32;
        constexpr int dk_dv_kBlockN = 128;
        constexpr int dk_dv_kBlockK = 32;
        constexpr int dk_dv_WARP_M = 32;
        constexpr int dk_dv_WARP_N = 32;
        dim3 dimBlock;
        int maxBlockThreads = 512;
        dimBlock.x = min((dk_dv_kBlockN)/(dk_dv_WARP_N)*64, maxBlockThreads);
        dimBlock.y = 1;
        dimBlock.z = 1;
        const int num_n_block = (params.seqlen_k + dk_dv_kBlockN - 1) / dk_dv_kBlockN;
        int gridDimx = num_n_block;
        // if constexpr (Is_causal){
        //     gridDimx = (num_n_block + 1) / 2;
        // }
        // if (params.deterministic) {
        //     gridDimx = min((params.cu_count * 2 + params.b * params.h - 1) / (params.b * params.h), gridDimx);
        // }
        // dim3 grid_n(gridDimx, params.h, params.b);
        dim3 grid_n(params.se_balance_cnt, gridDimx, (params.h * params.b/params.se_balance_cnt));
        // printf("flash_attention_dv_dk_bwd_kernel  : grid(%d, %d, %d) | block(%d, %d, %d)\n", grid_n.x, grid_n.y, grid_n.z, dimBlock.x, dimBlock.y, dimBlock.z);
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
                flash_attention_dv_dk_bwd_kernel<Element, float, Is_dropout, Is_causal, Is_local, IsEvenMNConst, IsEvenKConst, Is_first, Is_last, Seq_parallel,  dk_dv_kBlockM, dk_dv_kBlockN, K, K_v, dk_dv_kBlockK, dk_dv_WARP_M, dk_dv_WARP_N, USE_BSHD_LAYOUT>
                <<<grid_n, dimBlock, sharedMemSize, stream>>>(params);
            });
        });
    }

    {
         constexpr int dq_kBlockM = 128;
         constexpr int dq_kBlockN = 32;
         constexpr int dq_kBlockK = 32;
         constexpr int dq_WARP_M = 32;
         constexpr int dq_WARP_N = 32;
        int dq_kMThreads = (dq_kBlockM + dq_WARP_M-1)/dq_WARP_M * 64;
        const int num_m_block_dq = (params.seqlen_q + dq_kBlockM - 1) / dq_kBlockM;
        // dim3 grid_m(num_m_block_dq, params.h, params.b);
        dim3 grid_m(params.se_balance_cnt, num_m_block_dq, (params.h * params.b/params.se_balance_cnt));
        // printf("flash_attention_dq_bwd_kernel     : grid(%d, %d, %d) | block(%d, %d, %d)\n", grid_m.x, grid_m.y, grid_m.z, dq_kMThreads, 1, 1);
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
                flash_attention_dq_bwd_kernel<Element, float, Is_dropout, Is_causal, Is_local, IsEvenMNConst, IsEvenKConst, Is_first, Is_last, Seq_parallel,  dq_kBlockM, dq_kBlockN, K, K_v, dq_kBlockK, dq_WARP_M, dq_WARP_N, 2, USE_BSHD_LAYOUT>
                    <<<grid_m, dq_kMThreads, sharedMemSize, stream>>>(params);
            });
        });
    }
#if defined(FA_KERNEL_TIMER)
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipEventRecord(stop, 0 )) ;
    check_hip_error(hipEventSynchronize( stop ));

    float   ave_time;
    check_hip_error(hipEventElapsedTime( &ave_time,start, stop ));
    printf( "Time to generate:  %3.1f ms\n", ave_time );
    // f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    // return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)
    // std::size_t flop = (size_t(params.seqlen_q) * params.seqlen_k * K * 5) * params.b * params.h;
    size_t flop = 2.5 * 4 * params.b * params.seqlen_q * params.seqlen_k * params.h * K / (Is_causal ? 2: 1);
    printf("flop: %ld\n", flop);
    // std::size_t num_bytes =
    //     (sizeof(DataType) * seqlen_q * K + sizeof(DataType) * K * seqlen_k + sizeof(DataType) * seqlen_q * K ) *
    //     batch_size * num_heads;
    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    printf("tflops: \x1b[32m %f\033[0m\n", tflops);
    // float gb_per_sec = num_bytes / 1.E6 / ave_time;
#endif

    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        std::cout << "HIP 运行时 API 调用出错: " << hipGetErrorString(error) << std::endl;
    }
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    if (configure) return;
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal);
    BOOL_SWITCH(params.layout, USE_BSHD_LAYOUT, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            BOOL_SWITCH(is_swa, Is_local, [&] {
                run_flash_bwd_seqk_parallel<Is_causal && ! Is_local, !Is_causal && Is_local, Is_dropout,
                                            Kernel_traits,
                                            false,
                                            false,
                                            true,
                                            USE_BSHD_LAYOUT,
                                            Flash_bwd_params>(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 32;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, false>(params, stream, configure);
    }
}

template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 64;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, false>(params, stream, configure);
    }
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 96;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, true, T>, false>(params, stream, configure);
    }
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 128;
    if constexpr (std::is_same<T,Float8_e4m3_t>::value){
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 128, 64, 32, 32, 2, true, T>, false>(params, stream, configure);
    } else {
        if (params.p_dropout < 1.f) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 128, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 128, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
        }
    }
}

template<typename T>
void run_mha_bwd_hdim160(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 160;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
    }
}

template<typename T>
void run_mha_bwd_hdim192(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 192;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
    }
}


template<typename T>
void run_mha_bwd_qkhdim192_vhdim128(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int qkHeaddim = 192;
    constexpr int vHeaddim = 128;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<qkHeaddim, vHeaddim, 32, 32, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<qkHeaddim, vHeaddim, 32, 32, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
    }
}


template<typename T>
void run_mha_bwd_hdim224(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 224;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
    }
}

template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    constexpr int Headdim = 256;
    if (params.p_dropout < 1.f) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, true>(params, stream, configure);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 2, true, T>, false>(params, stream, configure);
    }
}
