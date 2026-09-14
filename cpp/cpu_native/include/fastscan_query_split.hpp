#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
namespace Chimera::cpu_kernel {
// Same 16-bit LUT, but process its low and high byte planes in separate passes.
// Each pass needs four accumulator vectors/query instead of eight.
// Packed codes are read and decoded twice; LUT bytes are each read once.
inline __m512i reduce_byte_accumulators(__m512i even, __m512i odd) {
    __m256i a = _mm256_add_epi16(_mm512_castsi512_si256(even), _mm512_extracti64x4_epi64(even, 1));
    __m256i b = _mm256_add_epi16(_mm512_castsi512_si256(odd), _mm512_extracti64x4_epi64(odd, 1));
    a = _mm256_sub_epi16(a, _mm256_slli_epi16(b, 8));
    return _mm512_add_epi32(
        _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(a, b, 0x21)),
        _mm512_cvtepu16_epi32(_mm256_blend_epi32(a, b, 0xf0)));
}
template<size_t Batch, size_t Byte, class Visitor>
inline void accumulate_byte_queries(const uint8_t* codes, const uint8_t* const* tables,
                                    size_t dim, Visitor&& visit) {
    __m512i acc[Batch][4];
    for (auto& q : acc) for (auto& a : q) a = _mm512_setzero_si512();
    const __m512i mask = _mm512_set1_epi8(15);
    for (size_t m=0; m<dim/4; m+=4) {
        __m512i c = _mm512_loadu_si512(codes + m*16);
        __m512i lo = _mm512_and_si512(c, mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c,4), mask);
        for (size_t q=0; q<Batch; ++q) {
            __m512i lut = _mm512_loadu_si512(tables[q] + (m/4)*128 + Byte*64);
            __m512i l = _mm512_shuffle_epi8(lut, lo);
            __m512i h = _mm512_shuffle_epi8(lut, hi);
            acc[q][0] = _mm512_add_epi16(acc[q][0], l);
            acc[q][1] = _mm512_add_epi16(acc[q][1], _mm512_srli_epi16(l,8));
            acc[q][2] = _mm512_add_epi16(acc[q][2], h);
            acc[q][3] = _mm512_add_epi16(acc[q][3], _mm512_srli_epi16(h,8));
        }
    }
    for (size_t q=0; q<Batch; ++q)
        visit(q, reduce_byte_accumulators(acc[q][0],acc[q][1]),
                 reduce_byte_accumulators(acc[q][2],acc[q][3]));
}
template<size_t Batch, class Visitor>
inline void accumulate_hacc_queries_split(const uint8_t* codes,
        const uint8_t* const* tables, size_t dim, Visitor&& visit) {
    __m512i low[Batch][2];
    accumulate_byte_queries<Batch,0>(codes,tables,dim,[&](size_t q,__m512i a,__m512i b){
        low[q][0]=a; low[q][1]=b;
    });
    accumulate_byte_queries<Batch,1>(codes,tables,dim,[&](size_t q,__m512i a,__m512i b){
        visit(q, _mm512_add_epi32(low[q][0],_mm512_slli_epi32(a,8)),
                 _mm512_add_epi32(low[q][1],_mm512_slli_epi32(b,8)));
    });
}
}
