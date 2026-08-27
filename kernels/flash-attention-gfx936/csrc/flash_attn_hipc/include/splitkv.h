#pragma once

#define PA_FIX_PARTITION 65536
#define MLA_FIX_PARTITION 65536
#define MLA_MAX_SPLITS 1024
#define MLA_MAX_SPLITS_INV 0.0009765625f
#define MLA_FIX_BALANCE_FACTOR 1.5f

template<int MIN_PARTITION_SIZE=128>
__forceinline__ __device__ int splitkv_get_partitionsize_of_fix_numsplits(int actual_seqlen_k, int num_splits) {
    float true_partition = max(1.f, actual_seqlen_k / float(num_splits));
    int partition_size = 1 << (int(log2f(true_partition - MLA_MAX_SPLITS_INV/*num_splits <= 1024*/)) + 1);
    while (num_splits * partition_size > MLA_FIX_BALANCE_FACTOR * actual_seqlen_k and num_splits * (partition_size - MIN_PARTITION_SIZE) > actual_seqlen_k)
        partition_size -= MIN_PARTITION_SIZE;
    partition_size = max(partition_size, MIN_PARTITION_SIZE);
    return partition_size;
}