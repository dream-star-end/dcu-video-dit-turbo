// Pytorch also has an implementation of Philox RNG: https://github.com/pytorch/pytorch/blob/8ca3c881db3e3510fcb7725389f6a0633c9b992c/torch/csrc/jit/tensorexpr/cuda_random.h
#pragma once
#include "numeric_types.h"

namespace flash {

struct ull2 {
    unsigned long long x;
    unsigned long long y;
};

__forceinline__ __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b) {
    uint2 res;
    asm ("v_mul_lo_u32 %0, %2, %3;\n\t"
		 "v_mul_hi_u32 %1, %2, %3;\n\t"
          : "=v"(res.x), "=v"(res.y)
          : "v"(a), "v"(b));
    __builtin_amdgcn_sched_barrier(0);
    return res;
}


__forceinline__ __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

__forceinline__ __device__ uint4 philox(unsigned long long seed,
                               unsigned long long subsequence,
                               unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ull2 *tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    uint4 output = philox_single_round(counter, key);
    return output;
}

// __forceinline__ __device__ uint4 philox(unsigned long long seed,
//                                                     unsigned long long subsequence,
//                                                     unsigned long long offset) {
//     unsigned long long x = seed ^ 0x9E3779B97F4A7C15ULL;
//     unsigned long long y = subsequence ^ 0xD1B54A32D192ED03ULL;
//     unsigned long long z = offset ^ 0x94D049BB133111EBULL;

//     x ^= (y + 0x9E3779B97F4A7C15ULL + (x << 6) + (x >> 2));
//     y ^= (z + 0xBF58476D1CE4E5B9ULL + (y << 6) + (y >> 2));
//     z ^= (x + 0x94D049BB133111EBULL + (z << 6) + (z >> 2));

//     uint4 out;
//     out.x = static_cast<uint32_t>(x);
//     out.y = static_cast<uint32_t>(x >> 32);
//     out.z = static_cast<uint32_t>(y ^ z);
//     out.w = static_cast<uint32_t>((y >> 32) ^ (z >> 32));
//     return out;
// }

// Position-addressed deterministic dropout RNG.
//
// The old helper packed (abs_q, abs_k) into an uint2 and then reinterpreted it
// as a 64-bit Philox offset.  That is fragile on device code and, more
// importantly for dropout replay, still couples the result to the exact Philox
// packing/consumption convention used by each path.  For dropout we only need a
// stable random pack for the logical attention coordinates.  Keep the mixer
// 32-bit only: 64-bit integer multiplies expand to a lot of VALU on DCU.
// The 2x2 helper below uses one 32-bit random value for four logical pixels, so
// forward QK and backward KQ can replay the same mask without physically
// transposing the tile.
__forceinline__ __device__ unsigned int dropout_rand_u32(
    unsigned long long seed,
    unsigned long long offset,
    int batch_head_idx,
    int abs_q,
    int abs_k)
{
    const uint2 seed32 = reinterpret_cast<const uint2&>(seed);
    const uint2 offset32 = reinterpret_cast<const uint2&>(offset);
    unsigned int x = seed32.x ^ 0x9E3779B9u;
    x ^= seed32.y + 0x85EBCA6Bu;
    x ^= offset32.x * 0xA24BAED5u;
    x ^= offset32.y + 0x9FB21C65u;
    x ^= static_cast<unsigned int>(batch_head_idx) * 0xC2B2AE35u;
    x ^= static_cast<unsigned int>(abs_q) * 0x27D4EB2Du;
    x ^= static_cast<unsigned int>(abs_k) * 0x165667B1u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__forceinline__ __device__ uint8_t philox_uniform_uint8(
    unsigned long long seed,
    unsigned long long offset,
    int batch_head_idx,
    int abs_q,
    int abs_k)
{
    return static_cast<uint8_t>(dropout_rand_u32(seed, offset, batch_head_idx, abs_q, abs_k) >> 24);
}

__forceinline__ __device__ unsigned int dropout_rand_2x2_u32(
    unsigned long long seed,
    unsigned long long offset,
    int batch_head_idx,
    int abs_q,
    int abs_k)
{
    return dropout_rand_u32(seed, offset, batch_head_idx, abs_q >> 1, abs_k >> 1);
}

__forceinline__ __device__ uint8_t dropout_rand_2x2_byte(
    unsigned int rand_pack,
    int abs_q,
    int abs_k)
{
    const int byte_idx = ((abs_q & 1) << 1) | (abs_k & 1);
    return static_cast<uint8_t>((rand_pack >> (byte_idx << 3)) & 0xffu);
}

__forceinline__ __device__ float encode_dropout_sign_bit(float x)
{
    union {
        float f;
        unsigned int u;
    } bits;
    bits.f = x;
    bits.u |= 0x80000000u;
    return bits.f;
}

__forceinline__ __device__ float clear_dropout_sign_bit(float x)
{
    union {
        float f;
        unsigned int u;
    } bits;
    bits.f = x;
    bits.u &= 0x7fffffffu;
    return bits.f;
}

} // namespace flash
