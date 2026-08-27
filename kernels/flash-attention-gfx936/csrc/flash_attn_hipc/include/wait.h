#ifndef WAIT_H
#define WAIT_H

#define USE_PINGPANG_BUFFER


namespace flash {

__forceinline__ __device__ void wait_all_warp_arrived() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_amdgcn_sched_barrier(0);
}


template<bool Sync>
__forceinline__ __device__ void wait_all_buffer_data_arrived() {
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n");
    } else {
        asm volatile("s_waitcnt vmcnt(0)\n");
    }
    __builtin_amdgcn_sched_barrier(0);
}


template<bool Sync>
__forceinline__ __device__ void wait_buffer_data_arrived(const int wait_count=0) {
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n":: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt vmcnt(%0)\n":: "n"(wait_count));
    }
    __builtin_amdgcn_sched_barrier(0);
}


template<bool Sync>
__forceinline__ __device__ void wait_lds_data_arrived(const int wait_count=0) {
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt lgkmcnt(%0)\n\ts_barrier\n":: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt lgkmcnt(%0)\n":: "n"(wait_count));
    }
    __builtin_amdgcn_sched_barrier(0);
}

} // namespace flash


template<const int COUNT>
__forceinline__ __device__ void buffer_load_lds_dwordx1_wait() {
    asm volatile(
      "s_waitcnt vmcnt(%0)\n\t"
      "s_barrier\n"
      :: "B"(COUNT)
      :);
}

template<const int COUNT>
__forceinline__ __device__ void buffer_load_lds_dwordx1_wait_nosync() {
    asm volatile(
      "s_waitcnt vmcnt(%0)\n\t"
      :: "B"(COUNT)
      :);
}


template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
inline __device__ void buffer_load_lds_dwordx1_wait() {
asm volatile("s_waitcnt vmcnt(0) \n\t"
                "s_barrier");
}

__forceinline__ __device__ void s_barrier() {
    asm volatile("s_barrier\n");
}

#define lgkmcnt_wait(X)\
__builtin_amdgcn_sched_barrier(0);\
asm volatile("s_waitcnt lgkmcnt(%0)": : "I"(X));\
__builtin_amdgcn_sched_barrier(0);

#define vmcnt_wait(X)\
__builtin_amdgcn_sched_barrier(0);\
    asm volatile(\
      "s_waitcnt vmcnt(%0)\n\t"\
      "s_barrier\n"\
      :: "I"(X)\
      :);\
__builtin_amdgcn_sched_barrier(0);   

#define vmcnt_wait_nosync(X)\
__builtin_amdgcn_sched_barrier(0);\
    asm volatile(\
      "s_waitcnt vmcnt(%0)\n\t"\
      :: "I"(X)\
      :);\
__builtin_amdgcn_sched_barrier(0);   

#endif
