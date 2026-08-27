

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_fwd_varlen_permute_bhsd2bshd(void* output, void* input, void* split_sizes, int num_heads, int real_headdim);

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_fwd_varlen_permute_bshd2bhsd(void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim);
