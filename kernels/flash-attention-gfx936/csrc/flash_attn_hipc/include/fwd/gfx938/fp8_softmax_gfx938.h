#pragma once

#include "philox.cuh"
#include "../utils.h"

using namespace flash;

template<int kBlockN, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_apply_mask(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    int max_seq_kv_offset,
    int wave_col_offset,
    int lane_id
) {
    __builtin_amdgcn_sched_barrier(0);
    const int col_base = wave_col_offset + (lane_id >> 4) * 8;
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        const int k_offset = k_loop * WARP_N;
        #pragma unroll
        for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
            const int n_base = col_base + n_idx * 4;
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32[vec_idx] =
                        (n_base + k_offset + vec_idx >= max_seq_kv_offset)
                            ? -INFINITY
                            : s_reg[k_loop][m_idx][n_idx].f32[vec_idx];
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<int kBlockN, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_apply_causal_mask(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    int actual_seqlen_q,
    int actual_seqlen_k,
    int wave_row_offset,
    int wave_col_offset,
    int lane_id
) {
    __builtin_amdgcn_sched_barrier(0);
    const int row_base = wave_row_offset + ((lane_id & 15) >> 2) * 8 + (lane_id & 3);
    const int col_base = wave_col_offset + (lane_id >> 4) * 8;
    const int causal_limit = actual_seqlen_k - actual_seqlen_q;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        const int row_idx = row_base + m_idx * 4;
        const int col_limit = min(actual_seqlen_k, row_idx + causal_limit);
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            const int k_offset = k_loop * WARP_N;
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                const int n_base = col_base + n_idx * 4;
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32[vec_idx] = (n_base + k_offset + vec_idx > col_limit) ? -INFINITY: s_reg[k_loop][m_idx][n_idx].f32[vec_idx];
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<int kBlockN, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_apply_local_mask(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    int actual_seqlen_q,
    int actual_seqlen_k,
    int wave_row_offset,
    int wave_col_offset,
    int window_size_left,
    int window_size_right,
    int lane_id
) {
    __builtin_amdgcn_sched_barrier(0);
    const int row_base = wave_row_offset + ((lane_id & 15) >> 2) * 8 + (lane_id & 3);
    const int col_base = wave_col_offset + (lane_id >> 4) * 8;
    const bool has_ws_left = window_size_left >= 0;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        const int row_idx = row_base + m_idx * 4;
        const int col_limit_left = max(0, row_idx + 1 + actual_seqlen_k - actual_seqlen_q - window_size_left);
        const int col_limit_right = min(actual_seqlen_k, row_idx + actual_seqlen_k - actual_seqlen_q + window_size_right);
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            const int k_offset = k_loop * WARP_N;
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                const int n_base = col_base + n_idx * 4;
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    const int col_idx = n_base + k_offset + vec_idx;
                    s_reg[k_loop][m_idx][n_idx].f32[vec_idx] =
                        (col_idx > col_limit_right || (has_ws_left && col_idx < col_limit_left - 1))
                            ? -INFINITY
                            : s_reg[k_loop][m_idx][n_idx].f32[vec_idx];
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<int kBlockN, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_qk_descale(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    __float2 qk_descale
) {
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                    s_reg[k_loop][m_idx][n_idx].u64[vec_idx] = __builtin_hcu_pk_mul_f32(s_reg[k_loop][m_idx][n_idx].u64[vec_idx], qk_descale);
                    // s_reg[k_loop][m_idx][n_idx].u64[vec_idx] = s_reg[k_loop][m_idx][n_idx].u64[vec_idx] * qk_descale;
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}


template<bool AssumeValidRows, int kHeadDim, int kBlockN, int WARP_M, int WARP_N, int WARP_K, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_softmax_and_schedule_v(
    /*softmax module related args*/
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum softmax_scale_log2,
    /*scheduled modules related args*/
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32],
    int8_t* v_lds
) {
    // ======================================================== Max ======================================================================
    ElementAccum scores_max_cur[WARP_M / 16];
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        ElementAccum max_value = scores_max[m_idx];
        // 当前线程遍历 4 个 32x32x32 mmac 输出的 f32x4
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    max_value = max(max_value, s_reg[k_loop][m_idx][n_idx].f32[vec_idx]);
                }
            }
        }
        // 这一行比较 0, 16, 32, 48 号线程的数据
        max_value = max(max_value, __shfl_xor_tmp(max_value, 32));
        max_value = max(max_value, __shfl_xor_tmp(max_value, 16));
        // 赋值给最终的最大值
        scores_max_cur[m_idx] = max_value;
    }

    // ========================================== softmax ===============================================
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        __float2 max_scaled_pair;
        if constexpr (AssumeValidRows) {
            max_scaled_pair[0] = -scores_max_cur[m_idx] * softmax_scale_log2;
        } else {
            max_scaled_pair[0] = scores_max_cur[m_idx] == -INFINITY ? 0.f: -scores_max_cur[m_idx] * softmax_scale_log2;
        }
        max_scaled_pair[1] = max_scaled_pair[0];
        __float2 softmax_scale_log2_pair = {softmax_scale_log2, softmax_scale_log2};
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                    s_reg[k_loop][m_idx][n_idx].u64[vec_idx] = __builtin_hcu_pk_fma_f32(s_reg[k_loop][m_idx][n_idx].u64[vec_idx], softmax_scale_log2_pair, max_scaled_pair);
                }
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32[vec_idx] = __llvm_exp2_f32(s_reg[k_loop][m_idx][n_idx].f32[vec_idx]);
                }
            }
        }
    }

    // ========================================== Sum ===============================================
    ElementAccum scores_sum_cur[WARP_M / 16];
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        vec2_Accum<ElementAccum> sum_pair;
        sum_pair.data = 0.0;
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                sum_pair.u64 = __builtin_hcu_pk_add_f32(sum_pair.u64, s_reg[k_loop][m_idx][n_idx].u64[0]);
                sum_pair.u64 = __builtin_hcu_pk_add_f32(sum_pair.u64, s_reg[k_loop][m_idx][n_idx].u64[1]);
            }
        }
        scores_sum_cur[m_idx] = sum_pair.f32[0] + sum_pair.f32[1];
        scores_sum_cur[m_idx] = scores_sum_cur[m_idx] + __shfl_xor_tmp(scores_sum_cur[m_idx], 32);
        scores_sum_cur[m_idx] = scores_sum_cur[m_idx] + __shfl_xor_tmp(scores_sum_cur[m_idx], 16);
    }

    // 更新 scores_sum, scores_max
    // 这段代码放在这是因为即将下发的大量 ds 指令, 会跟 __shfl_xor 抢带宽, 导致时延太高
    // ElementAccum exp_rescale[WARP_M / 16];
    // #pragma unroll
    // for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
    //     exp_rescale[m_idx] = __llvm_exp2_f32((scores_max[m_idx] - scores_max_cur[m_idx]) * softmax_scale_log2);
    //     scores_max[m_idx]  = scores_max_cur[m_idx];
    //     scores_sum[m_idx]  = __llvm_fma_f32(scores_sum[m_idx], exp_rescale[m_idx], scores_sum_cur[m_idx]);
    // }

    // ========================================== schedule V ===============================================
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n");
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        // 用 ds_read_matrix 从 lds 读取数据到寄存器
        int8_t* lds_load_ptr = v_lds + k_loop * WARP_M * kHeadDim * sizeof(Element);
        v_regs[k_loop][0].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr, 0, 2, 2, 0);
        v_regs[k_loop][1].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 32, 0, 2, 2, 0);
        v_regs[k_loop][2].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 128 * 16, 0, 2, 2, 0);
        v_regs[k_loop][3].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 128 * 16 + 32, 0, 2, 2, 0);
        if constexpr (kHeadDim == 256) {
            v_regs[k_loop][4].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 4096, 0, 2, 2, 0);
            v_regs[k_loop][5].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 4096 + 32, 0, 2, 2, 0);
            v_regs[k_loop][6].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 4096 + 128 * 16, 0, 2, 2, 0);
            v_regs[k_loop][7].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(lds_load_ptr + 4096 + 128 * 16 + 32, 0, 2, 2, 0);
        }
    }
    __builtin_amdgcn_sched_barrier(0); // hint: 这里考虑只发一部分的 ds_read_matrix 指令出去, 一面堵住

    // ========================================== rescale ===============================================
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; m_idx += 1) {
        if (scores_sum[m_idx] != 0.f && scores_max[m_idx] < scores_max_cur[m_idx]) {
            __float2 scores_scale_pair;
            float max_diff;
            if constexpr (AssumeValidRows) {
                max_diff = scores_max[m_idx] - scores_max_cur[m_idx];
            } else {
                // Fix: 当 scores_max 和 scores_max_cur 都是 -INFINITY 时，(-INF) - (-INF) = NaN
                // 这种情况发生在某些 query 行完全没有有效的 KV 可以 attend 时
                max_diff = (scores_max[m_idx] == -INFINITY || scores_max_cur[m_idx] == -INFINITY)
                             ? 0.f : (scores_max[m_idx] - scores_max_cur[m_idx]);
            }
            scores_scale_pair[0] = __llvm_exp2_f32(max_diff * softmax_scale_log2);
            scores_scale_pair[1] = scores_scale_pair[0];
            scores_sum[m_idx] *= scores_scale_pair[0];
            // 放缩 acc_o
            #pragma unroll
            for (int pv_loop = 0; pv_loop < kHeadDim / WARP_N; ++pv_loop) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_K / 16; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[pv_loop][m_idx][mmac_id].u64[0], scores_scale_pair);
                    acc_o[pv_loop][m_idx][mmac_id].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[pv_loop][m_idx][mmac_id].u64[1], scores_scale_pair);
                }
            }
        }
    }

    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        scores_max[m_idx] = scores_max_cur[m_idx];
        scores_sum[m_idx] += scores_sum_cur[m_idx];
    }
}




template<int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_cvt_f32_to_fp8(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16]
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                __builtin_hcu_cvt_pk4_fp8_f32<Element>(s_reg[k_loop][m_idx][n_idx].f32, p_reg[m_idx].i32[k_loop * 2 + n_idx]);
            }
        }
    }
}
