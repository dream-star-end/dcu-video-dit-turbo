
#define DEFINE_SPLITKV_MEMORY_MANAGER \
class SplitkvMemoryManager { \
private: \
    int num_splits, batch_size, num_heads, seqlen_q, head_size; \
    bool is_split_empty = true; \
public: \
    at::Tensor scores_sum, scores_max, out_accum; \
public: \
    void set_split_memory( \
        int __num_splits, int __batch_size, int __num_heads, \
        int __seqlen_q, int __head_size, const torch::TensorOptions opts) { \
        if (!this->is_split_empty and this->is_same_split(__num_splits, __batch_size, __num_heads, __seqlen_q, __head_size)) { \
        } \
        else if (!this->is_split_empty and this->is_compatible_split(__num_splits, __batch_size, __num_heads, __seqlen_q, __head_size)) { \
            this->scores_sum = this->scores_sum.view({__num_splits, __batch_size, __num_heads, __seqlen_q}).contiguous(); \
            this->scores_max = this->scores_max.view({__num_splits, __batch_size, __num_heads, __seqlen_q}).contiguous(); \
            this->out_accum  = this->out_accum.view({__num_splits, __batch_size, __num_heads, __seqlen_q, __head_size}).contiguous(); \
        } \
        else { \
            auto raw_memory  = torch::empty({2, __num_splits, __batch_size, __num_heads, __seqlen_q}, opts.dtype(at::kFloat)); \
            this->scores_sum = raw_memory.index({0}); \
            this->scores_max = raw_memory.index({1}); \
            this->out_accum  = torch::empty({__num_splits, __batch_size, __num_heads, __seqlen_q, __head_size}, opts.dtype(at::kHalf)); \
            this->num_splits = __num_splits; \
            this->batch_size = __batch_size; \
            this->num_heads  = __num_heads; \
            this->seqlen_q   = __seqlen_q; \
            this->head_size  = __head_size; \
            this->is_split_empty = false; \
        } \
    } \
    bool is_same_split(int __num_splits, int __batch_size, int __num_heads, int __seqlen_q, int __head_size) { \
        return (this->num_splits == __num_splits) and (this->batch_size == __batch_size) \
            and (this->num_heads == __num_heads) and (this->seqlen_q == __seqlen_q) and (this->head_size == __head_size); \
    } \
    bool is_compatible_split(int __num_splits, int __batch_size, int __num_heads, int __seqlen_q, int __head_size) { \
        return (__num_splits * __batch_size * __num_heads * __seqlen_q * __head_size) \
            == (this->num_splits * this->batch_size * this->num_heads * this->seqlen_q * this->head_size); \
    } \
};