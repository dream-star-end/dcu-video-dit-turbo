// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once
#include "numeric_types.h"
#include "splitkv.h"

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = Float16;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = BFloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define LAYOUT_SWITCH(layout, ...)         \
  [&] {                                    \
    if (layout == 0) {                     \
      constexpr static int Layout = 0;     \
      return __VA_ARGS__();                \
    } else if (layout == 1) {              \
      constexpr static int Layout = 1;     \
      return __VA_ARGS__();                \
    }                                      \
  }()

// #define ElementType_SWITCH(is_bf16, is_e4m3, ...)   \
//   [&] {                                      \
//     if (is_bf16) {                           \
//       using elem_type = BFloat16;            \
//       return __VA_ARGS__();                  \
//     } else if (is_e4m3) {                    \
//       using elem_type = Float8_e4m3_t;       \
//       return __VA_ARGS__();                  \
//     } else {                                 \
//       using elem_type = Float16;             \
//       return __VA_ARGS__();                  \
//     }                                        \
//   }()

#define ElementType_SWITCH(is_bf16, is_e4m3, ...)   \
[&] {                                      \
  if (is_bf16) {                           \
    using elem_type = BFloat16;            \
    return __VA_ARGS__();                  \
  } else if (is_e4m3) {                    \
    printf("fa bwd does not support fp8 yet");\
  } else {                                 \
    using elem_type = Float16;             \
    return __VA_ARGS__();                  \
  }                                        \
}()

#define HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 256) {             \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define ALL_HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    }  else if (HEADDIMQ <= 256) {            \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 512) {             \
      constexpr static int kHeadDimQ = 512;   \
      constexpr static int kHeadDimV = 512;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define WARP_ID_SWITCH(warp_id, ...)   \
  [&] {                                    \
    if (warp_id == 0) {                    \
      constexpr  int WARP_ID = 0;          \
      return __VA_ARGS__();                \
    } else if (warp_id == 1) {             \
      constexpr  int WARP_ID = 1;          \
      return __VA_ARGS__();                \
    } else if (warp_id == 2) {             \
      constexpr  int WARP_ID = 2;          \
      return __VA_ARGS__();                \
    } else if (warp_id == 3) {             \
      constexpr  int WARP_ID = 3;          \
      return __VA_ARGS__();                \
    }                                      \
  }()


#define CU_SWITCH(cu_count, ...)           \
  [&] {                                    \
    if (cu_count == 120) {                 \
      constexpr  int CU_COUNT = 120;       \
      return __VA_ARGS__();                \
    } else if (cu_count == 128) {          \
      constexpr  int CU_COUNT = 128;       \
      return __VA_ARGS__();                \
    } else if (cu_count == 88) {           \
      constexpr  int CU_COUNT = 88;        \
      return __VA_ARGS__();                \
    } else if (cu_count == 80) {           \
      constexpr  int CU_COUNT = 80;        \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define M_MMAC_COUNT_SWITCH(COND, M_MMAC_COUNT, ...)    \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static int M_MMAC_COUNT = 2;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int M_MMAC_COUNT = 1;    \
      return __VA_ARGS__();                     \
    }                                           \
  }()



#define REUSEKV_SWITCH(seqlen_q , ...)                  \
[&] {                                                   \
    if (seqlen_q == 16) {                               \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 13) {                        \
        constexpr static int REUSE_KV_TIMES = 13;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 8) {                         \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 7) {                         \
        constexpr static int REUSE_KV_TIMES = 7;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 4) {                         \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 3) {                         \
        constexpr static int REUSE_KV_TIMES = 3;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 2) {                         \
        constexpr static int REUSE_KV_TIMES = 2;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 1) {                         \
        constexpr static int REUSE_KV_TIMES = 1;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 29) {                        \
        constexpr static int REUSE_KV_TIMES = 29;       \
        return __VA_ARGS__();                           \
    } else {                                            \
        constexpr static int REUSE_KV_TIMES = 0;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()


#define PA_GQA_REGROUP_SWITCH(ngroups, ...) \
[&] {                               \
    if (ngroups == 8) {             \
        int GQA_REGROUP = 8;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 4) {      \
        int GQA_REGROUP = 4;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 2) {      \
        int GQA_REGROUP = 2;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 9) {      \
        int GQA_REGROUP = 9;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 7) {      \
        int GQA_REGROUP = 7;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 5) {      \
        int GQA_REGROUP = 5;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 3) {      \
        int GQA_REGROUP = 3;        \
        return __VA_ARGS__();       \
    } else if (ngroups == 29) {     \
        int GQA_REGROUP = 29;       \
        return __VA_ARGS__();       \
    } else {                        \
        int GQA_REGROUP = ngroups;  \
        return __VA_ARGS__();       \
    }                               \
}()

#define PA_MTP_REUSEKV_SWITCH(pa_mtp , ...)             \
[&] {                                                   \
    if (pa_mtp == 8) {                                  \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 16) {                          \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 20) {                          \
        constexpr static int REUSE_KV_TIMES = 20;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 24) {                          \
        constexpr static int REUSE_KV_TIMES = 24;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 28) {                          \
        constexpr static int REUSE_KV_TIMES = 28;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 32) {                          \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    } else {                                            \
        constexpr static int REUSE_KV_TIMES = 0;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()


#define MLA_REUSEKV_SWITCH(seqlen_q , ...)              \
[&] {                                                   \
    if (seqlen_q == 128) {                              \
        constexpr static int REUSE_KV_TIMES = 128;      \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 64) {                        \
        constexpr static int REUSE_KV_TIMES = 64;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 32) {                        \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 16) {                        \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 12) {                        \
        constexpr static int REUSE_KV_TIMES = 12;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 8) {                         \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 4) {                         \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 2) {                         \
        constexpr static int REUSE_KV_TIMES = 2;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 1) {                         \
        constexpr static int REUSE_KV_TIMES = 1;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()


#define MLA_MTP_REUSEKV_SWITCH(seqlen_q , ...)          \
[&] {                                                   \
    if (seqlen_q == 32) {                               \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 24) {                        \
        constexpr static int REUSE_KV_TIMES = 24;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 16) {                        \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else {                                            \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()


#define PERMUTE_DWORD_SWITCH(COND, DWORD_PER_TX, ...)    \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static int DWORD_PER_TX = 4;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int DWORD_PER_TX = 1;    \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define PERMUTE_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM == 64) {                   \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM < 256) {            \
      constexpr static int kHeadDim = 0;   \
      return __VA_ARGS__();                \
    }  else if (HEADDIM == 256) {          \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()


#define PA_HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 256) {             \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ == 512) {             \
      constexpr static int kHeadDimQ = 512;   \
      constexpr static int kHeadDimV = 512;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ == 576) {             \
      constexpr static int kHeadDimQ = 576;   \
      constexpr static int kHeadDimV = 512;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define PA_PAGEBLOCKSIZE_SWITCH(page_block_size, ...)                      \
  [&] {                                                                    \
    if (page_block_size % 32 == 0 and page_block_size % 64 != 0) {         \
      constexpr static int kBlockN = 32;                                   \
      return __VA_ARGS__();                                                \
    } else if (page_block_size % 64 == 0 and page_block_size % 128 != 0) { \
      constexpr static int kBlockN = 64;                                   \
      return __VA_ARGS__();                                                \
    } else if (page_block_size % 128 == 0) {                               \
      constexpr static int kBlockN = 128;                                  \
      return __VA_ARGS__();                                                \
    }                                                                      \
  }()


#define MLA_PARTITION_SIZE_SWITCH(partition_size, num_splits, ...)         \
  [&] {                                                                    \
    if (partition_size == 128 and num_splits > 1) {                        \
      constexpr static int Partition_Size = 1;                             \
      return __VA_ARGS__();                                                \
    } else if (partition_size == 256 and num_splits > 1) {                 \
      constexpr static int Partition_Size = 2;                             \
      return __VA_ARGS__();                                                \
    } else if (partition_size == 512 and num_splits > 1) {                 \
      constexpr static int Partition_Size = 4;                             \
      return __VA_ARGS__();                                                \
    } else if (partition_size == 1024 and num_splits > 1) {                \
      constexpr static int Partition_Size = 8;                             \
      return __VA_ARGS__();                                                \
    } else if (partition_size == MLA_FIX_PARTITION and num_splits > 1) {   \
      constexpr static int Partition_Size = MLA_FIX_PARTITION;             \
      return __VA_ARGS__();                                                \
    } else {                                                               \
      constexpr static int Partition_Size = 0;                             \
      return __VA_ARGS__();                                                \
    }                                                                      \
  }()
