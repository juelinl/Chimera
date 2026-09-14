#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
namespace Chimera::cpu_kernel {
// True register tile: Q query vectors by B packed blocks (32 document tokens each).
// A code load is shared across Q; a LUT load is shared across B.
template<size_t Q,size_t B,class Visitor>
inline void accumulate_hacc_tile(const uint8_t* codes, const uint8_t* const* tables,
                                size_t dim, Visitor&& visit) {
    __m512i accumulators[Q][B][2][4];
    for(auto& q:accumulators)for(auto& b:q)for(auto& p:b)for(auto& a:p)
        a=_mm512_setzero_si512();
    const __m512i mask=_mm512_set1_epi8(15);
    for(size_t m=0;m<dim/4;m+=4) {
        __m512i lo[B],hi[B];
        for(size_t b=0;b<B;++b) {
            __m512i c=_mm512_loadu_si512(codes+b*dim*4+m*16);
            lo[b]=_mm512_and_si512(c,mask);
            hi[b]=_mm512_and_si512(_mm512_srli_epi16(c,4),mask);
        }
        for(size_t q=0;q<Q;++q)for(size_t plane=0;plane<2;++plane) {
            __m512i lut=_mm512_loadu_si512(tables[q]+(m/4)*128+plane*64);
            for(size_t b=0;b<B;++b) {
                auto& a=accumulators[q][b][plane];
                __m512i l=_mm512_shuffle_epi8(lut,lo[b]);
                __m512i h=_mm512_shuffle_epi8(lut,hi[b]);
                a[0]=_mm512_add_epi16(a[0],l);
                a[1]=_mm512_add_epi16(a[1],_mm512_srli_epi16(l,8));
                a[2]=_mm512_add_epi16(a[2],h);
                a[3]=_mm512_add_epi16(a[3],_mm512_srli_epi16(h,8));
            }
        }
    }
    // std::cerr << "FastScan YES!" << std::endl;

    for (size_t q = 0; q < Q; ++q) for(size_t b=0;b<B;++b) {
    auto& accu = accumulators[q][b];
    __m512i res[2];
    __m512i dis0[2];
    __m512i dis1[2];

    for (size_t i = 0; i < 2; ++i) {
        __m256i tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][0]), _mm512_extracti64x4_epi64(accu[i][0], 1)
        );
        __m256i tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][1]), _mm512_extracti64x4_epi64(accu[i][1], 1)
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));

        dis0[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xF0))
        );

        __m256i tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][2]), _mm512_extracti64x4_epi64(accu[i][2], 1)
        );
        __m256i tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][3]), _mm512_extracti64x4_epi64(accu[i][3], 1)
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));

        dis1[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xF0))
        );
    }
    // shift res of high, add res of low
    res[0] =
        _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8));  // res for vec 0 to 15
    res[1] =
        _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8));  // res for vec 16 to 31

    visit(q,b,res[0],res[1]);
    }
}
}
