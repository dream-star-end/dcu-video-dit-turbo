

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_permute_sbhd2bhsd(void* output, void* input, int32_t seqlen, int real_headdim);

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_permute_bhsd2sbhd(void* output, void* input, int32_t seqlen, int real_headdim);

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_permute_bshd2bhsd(void* output, void* input, int32_t seqlen, int32_t num_heads);

template<int32_t kHeadDim, int32_t dword_count, int32_t seqlen_per_block=0>
__global__ void flash_permute_bhsd2bshd(void* output, void* input, int32_t seqlen, int32_t num_heads);

