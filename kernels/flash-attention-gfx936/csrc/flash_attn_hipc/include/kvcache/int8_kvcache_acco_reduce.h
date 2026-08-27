#include "numeric_types.h"


template<int REUSE_KV_TIMES, int kHeadDim, int kBlockK, int WARP_M, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void int8_kvcache_acco_reduce(
        vec4_Accum<ElementAccum> acc_o[(kHeadDim/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int WARP_ID,
        int lane_id) {

    // when REUSE_KV not in templated, compute max reuse times
    int EVEN_REUSE_KV_TIMES = (REUSE_KV_TIMES > 0) ? ((REUSE_KV_TIMES + 1) / 2) * 2: ((seqlen_q + 1) / 2) * 2;
    int HALF_REUSE_KV_TIMES = EVEN_REUSE_KV_TIMES >> 1;

    int q_seq_idx = (lane_id & 15);
    constexpr int __kHeadDim = (REUSE_KV_TIMES >= 16) ? kHeadDim: kHeadDim + 4/*<=15 can use misalign to reduce bank conflicts, but >16 may lead to lds>32KB, less waves per SIMD*/;
    if (q_seq_idx < HALF_REUSE_KV_TIMES) {
        // ####################################################################################################################################################
        // 4 个 wave 分别把自己负责的 acc_o 计算结果写到 LDS 中
        for(int h_idx=0; h_idx<(kHeadDim/kBlockK); h_idx++) {
            for(int k_idx=0; k_idx<(kBlockK/32); k_idx++) {
                for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        int lds_offset = WARP_ID*EVEN_REUSE_KV_TIMES*__kHeadDim + q_seq_idx*2*__kHeadDim + min_tile_m*__kHeadDim + h_idx*kBlockK + k_idx*32 + min_tile_k*16 + (lane_id>>4)*4;
                        *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[h_idx* ((WARP_M/32)*(kBlockK/32)) + k_idx*(WARP_M/32)][min_tile_k*2 + min_tile_m].f32;
                    }
                }
            }
        }
        __syncthreads();
        // ####################################################################################################################################################
        // 4 个 wave 共同参与 acc_o 在 LDS 中的相加
        // 判断当前架构是否支持 pk_f32 指令
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            constexpr bool SUPPORT_PK_F32 = true;
        #else
            constexpr bool SUPPORT_PK_F32 = false;
        #endif
        // 当 gfx936 而且 EVEN_REUSE_KV_TIMES 编译期可知的情况下, 可以大胆使用 ds_read2_b32 指令; 且是 GQA 的情况下, 可以更好地使用 pk 指令, 直接内联汇编控制
        if constexpr (SUPPORT_PK_F32 and REUSE_KV_TIMES > 0 and M_MMAC_COUNT > 1 and kHeadDim == 128) {
            // static_assert (kBlockK  == 32  && "only kBlockK=32 is supported!");
            // static_assert (kHeadDim == 128 && "only kHeadDim=128 is supported!");
            union_vec2_fp32 acc_tmp_wave0[4*2];
            union_vec2_fp32 acc_tmp_wave1[2], acc_tmp_wave2[2], acc_tmp_wave3[2];

            // 先预取第一次数据
            int loop_id = 0;
            int lds_offset[4*2];
            lds_offset[0] = 0*__kHeadDim + q_seq_idx*2*__kHeadDim + 0*kBlockK + (lane_id>>4)*4 + WARP_ID;
            inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id], acc_tmp_wave0[loop_id].u64, 0, 16);
            inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[0].u64, 0, 16);
            inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[0].u64, 0, 16);
            inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[0].u64, 0, 16);
            __builtin_amdgcn_sched_barrier(0);
            // 先计算 lds_offset
            lds_offset[1] = 1*__kHeadDim + q_seq_idx*2*__kHeadDim + 0*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[2] = 0*__kHeadDim + q_seq_idx*2*__kHeadDim + 1*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[3] = 1*__kHeadDim + q_seq_idx*2*__kHeadDim + 1*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[4] = 0*__kHeadDim + q_seq_idx*2*__kHeadDim + 2*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[5] = 1*__kHeadDim + q_seq_idx*2*__kHeadDim + 2*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[6] = 0*__kHeadDim + q_seq_idx*2*__kHeadDim + 3*kBlockK + (lane_id>>4)*4 + WARP_ID;
            lds_offset[7] = 1*__kHeadDim + q_seq_idx*2*__kHeadDim + 3*kBlockK + (lane_id>>4)*4 + WARP_ID;
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 0/*h_idx*2 + min_tile_m*/;

                // 预取下一阶段的数据
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[1].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[0].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[0].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[0].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 1/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[0].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[1].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[1].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[1].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 2/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[1].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[0].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[0].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[0].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 3/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[0].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[1].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[1].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[1].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 4/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[1].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[0].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[0].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[0].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 5/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[0].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[0].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[1].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[1].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[1].u64);
            }
            // asm volatile("s_nop 8\n");
            {
                int loop_id = 6/*h_idx*2 + min_tile_m*/;

                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1], acc_tmp_wave0[loop_id + 1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave1[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave2[1].u64, 0, 16);
                inlineasm_fa_ds_read2_b32(acc_o_lds, lds_offset[loop_id + 1] + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, acc_tmp_wave3[1].u64, 0, 16);

                asm volatile("s_waitcnt lgkmcnt(6)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[0].u64);
                asm volatile("s_waitcnt lgkmcnt(5)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[0].u64);
                asm volatile("s_waitcnt lgkmcnt(4)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[0].u64);
            }
            // 先写一部分数据到 lds
            for (int loop_id = 0; loop_id < 7; ++loop_id) {
                acc_o_lds[lds_offset[loop_id]]      = acc_tmp_wave0[loop_id].f32[0];
                acc_o_lds[lds_offset[loop_id] + 16] = acc_tmp_wave0[loop_id].f32[1];
            }
            // 再等待最后一部分需要的数据回来, 计算和写最后的数据
            {
                int loop_id = 7/*h_idx*2 + min_tile_m*/;

                asm volatile("s_waitcnt lgkmcnt(2)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave1[1].u64);
                asm volatile("s_waitcnt lgkmcnt(1)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave2[1].u64);
                asm volatile("s_waitcnt lgkmcnt(0)\n");
                __builtin_amdgcn_sched_barrier(0);
                acc_tmp_wave0[loop_id].u64 = __builtin_hcu_pk_add_f32(acc_tmp_wave0[loop_id].u64, acc_tmp_wave3[1].u64);
                __builtin_amdgcn_sched_barrier(0);
                acc_o_lds[lds_offset[loop_id]]      = acc_tmp_wave0[loop_id].f32[0];
                acc_o_lds[lds_offset[loop_id] + 16] = acc_tmp_wave0[loop_id].f32[1];
                // inline_ds_write2_b32_no_wait_bytes(acc_o_lds, lds_offset[loop_id], acc_tmp_wave0[loop_id].u64, 0, 16);
            }
            // 代替 __syncthreads()
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_barrier");
            __builtin_amdgcn_sched_barrier(0);
        }
        // 当 EVEN_REUSE_KV_TIMES 编译期可知的情况下, 可以大胆使用 ds_read2_b32 指令, gfx928 和 gfx936 都能用
        else if constexpr (REUSE_KV_TIMES > 0) {
            for(int h_idx=0; h_idx<(kHeadDim/kBlockK); h_idx++) {
                for(int k_idx=0; k_idx<(kBlockK/32); k_idx++) {
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        union_vec2_fp32 acc_tmp;
                        int lds_offset0 = min_tile_m*__kHeadDim + q_seq_idx*2*__kHeadDim + h_idx*kBlockK + k_idx*32 + 0*16 + (lane_id>>4)*4 + WARP_ID;
                        int lds_offset1 = min_tile_m*__kHeadDim + q_seq_idx*2*__kHeadDim + h_idx*kBlockK + k_idx*32 + 1*16 + (lane_id>>4)*4 + WARP_ID;
                        acc_tmp.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0, 0, 16, false);
                        // acc_tmp.f32[0]  = acc_o_lds[lds_offset0];
                        // acc_tmp.f32[1]  = acc_o_lds[lds_offset1];
                        union_vec2_fp32 acc_tmp_wave1;
                        acc_tmp_wave1.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 1*EVEN_REUSE_KV_TIMES*__kHeadDim, 0, 16, false);
                        // acc_tmp_wave1.f32[0] = acc_o_lds[lds_offset0 + 1*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        // acc_tmp_wave1.f32[1] = acc_o_lds[lds_offset1 + 1*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        acc_tmp.f32[0] += acc_tmp_wave1.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave1.f32[1];
                        union_vec2_fp32 acc_tmp_wave2;
                        acc_tmp_wave2.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 2*EVEN_REUSE_KV_TIMES*__kHeadDim, 0, 16, false);
                        // acc_tmp_wave2.f32[0] = acc_o_lds[lds_offset0 + 2*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        // acc_tmp_wave2.f32[1] = acc_o_lds[lds_offset1 + 2*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        acc_tmp.f32[0] += acc_tmp_wave2.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave2.f32[1];
                        union_vec2_fp32 acc_tmp_wave3;
                        acc_tmp_wave3.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 3*EVEN_REUSE_KV_TIMES*__kHeadDim, 0, 16, false);
                        // acc_tmp_wave3.f32[0] = acc_o_lds[lds_offset0 + 3*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        // acc_tmp_wave3.f32[1] = acc_o_lds[lds_offset1 + 3*EVEN_REUSE_KV_TIMES*__kHeadDim];
                        acc_tmp.f32[0] += acc_tmp_wave3.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave3.f32[1];
                        // ds_write2_b32
                        acc_o_lds[lds_offset0] = acc_tmp.f32[0];
                        acc_o_lds[lds_offset1] = acc_tmp.f32[1];
                    }
                }
            }
            __syncthreads();
        } else {
            // REUSE_KV_TIMES 编译期不可知, 导致 EVEN_REUSE_KV_TIMES 也编译期不可知, 无法直接调用 ds_read2_b32, 所以交给编译器去优化
            for(int h_idx=0; h_idx<(kHeadDim/kBlockK); h_idx++) {
                for(int k_idx=0; k_idx<(kBlockK/32); k_idx++) {
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                            int lds_offset = min_tile_m*__kHeadDim + q_seq_idx*2*__kHeadDim + h_idx*kBlockK + k_idx*32 + min_tile_k*16 + (lane_id>>4)*4 + WARP_ID;
                            float acc_tmp_wave0  = acc_o_lds[lds_offset];
                            for(int loop=1; loop<4; loop++) {
                                acc_tmp_wave0 += acc_o_lds[lds_offset + loop*EVEN_REUSE_KV_TIMES*__kHeadDim];
                            }
                            acc_o_lds[lds_offset] = acc_tmp_wave0;
                        }
                    }
                }
            }
            __syncthreads();
        }
        // ####################################################################################################################################################
        // 每个 wave 都从 LDS 获取最终的求和结果
        for(int h_idx=0; h_idx<(kHeadDim/kBlockK); h_idx++) {
            for(int k_idx=0; k_idx<(kBlockK/32); k_idx++) {
                for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        int lds_offset = q_seq_idx*2*__kHeadDim + min_tile_m*__kHeadDim + h_idx*kBlockK + k_idx*32 + min_tile_k*16 + (lane_id>>4)*4;
                        acc_o[h_idx* ((WARP_M/32)*(kBlockK/32)) + k_idx*(WARP_M/32)][min_tile_k*2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds + lds_offset);
                    }
                }
            }
        }
    }
}