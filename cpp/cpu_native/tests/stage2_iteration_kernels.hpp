#pragma once
#include "fastscan_query_split.hpp"
// Experimental scorers used only by the microbenchmark and correctness tests.
namespace Chimera::cpu_kernel {
inline __m512 late_broadcast(const float& x) {
    __m512 out;
    // Keep invariant broadcasts out of the integer loop's live vector set.
    // Volatile requires this load at each epilogue; no rounding/precision change.
    asm volatile("vbroadcastss %1, %0" : "=v"(out) : "m"(x));
    return out;
}
template<size_t Batch, bool Split, bool Late, class Cache>
__attribute__((noinline)) float stage2_score_doc_experiment(
    const Cache& cache, size_t dim,
    const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
    const query_object* queries, size_t nq, size_t start, size_t length) {
    if constexpr (Batch != 1) {
        if (nq % Batch)
            return stage2_score_doc_experiment<1,Split,Late>(cache,dim,luts,queries,nq,start,length);
    }
    const size_t first_batch=start/32,last_batch=(start+length-1)/32;
    const float* factors=cache.factors();
    float score=0;
    for(size_t first=0;first<nq;first+=Batch) {
        const uint8_t* tables[Batch];
        float delta[Batch],offset[Batch],bias[Batch];
        __m512 maxima[Batch][2];
        for(size_t q=0;q<Batch;++q) {
            tables[q]=luts[first+q]->lut();
            delta[q]=luts[first+q]->delta();
            offset[q]=luts[first+q]->sum_vl();
            bias[q]=queries[first+q].cb1_sumq;
            maxima[q][0]=maxima[q][1]=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
        }
        for(size_t b=first_batch;b<=last_batch;++b) {
            size_t begin=std::max(start,b*32)-b*32;
            size_t end=std::min(start+length,b*32+32)-b*32;
            uint32_t mask=uint32_t((uint64_t(1)<<end)-1)&~uint32_t((uint64_t(1)<<begin)-1);
            auto epilogue=[&](size_t q,__m512i a0,__m512i a1) {
                __m512 d,o,c;
                if constexpr(Late) {
                    d=late_broadcast(delta[q]);o=late_broadcast(offset[q]);c=late_broadcast(bias[q]);
                } else {
                    d=_mm512_set1_ps(delta[q]);o=_mm512_set1_ps(offset[q]);c=_mm512_set1_ps(bias[q]);
                }
                __m512 v0=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(d,_mm512_cvtepi32_ps(a0),o),c),
                                        _mm512_loadu_ps(factors+b*32));
                __m512 v1=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(d,_mm512_cvtepi32_ps(a1),o),c),
                                        _mm512_loadu_ps(factors+b*32+16));
                maxima[q][0]=_mm512_mask_max_ps(maxima[q][0],__mmask16(mask),maxima[q][0],v0);
                maxima[q][1]=_mm512_mask_max_ps(maxima[q][1],__mmask16(mask>>16),maxima[q][1],v1);
            };
            if constexpr(Split) accumulate_hacc_queries_split<Batch>(cache.batch_codes(b),tables,dim,epilogue);
            else accumulate_hacc_queries_visit<Batch>(cache.batch_codes(b),tables,dim,epilogue);
        }
        for(size_t q=0;q<Batch;++q)
            score+=_mm512_reduce_max_ps(_mm512_max_ps(maxima[q][0],maxima[q][1]));
    }
    return score;
}
// Fold both halves into one maximum vector. Valid finite scores preserve MaxSim.
template<size_t Batch, bool Split, bool Late, class Cache>
__attribute__((noinline)) float stage2_score_doc_compact(
    const Cache& cache, size_t dim,
    const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
    const query_object* queries, size_t nq, size_t start, size_t length) {
    if constexpr (Batch != 1) {
        if (nq % Batch)
            return stage2_score_doc_compact<1,Split,Late>(cache,dim,luts,queries,nq,start,length);
    }
    const size_t first_batch=start/32,last_batch=(start+length-1)/32;
    const float* factors=cache.factors();
    float score=0;
    for(size_t first=0;first<nq;first+=Batch) {
        const uint8_t* tables[Batch];
        float delta[Batch],offset[Batch],bias[Batch];
        __m512 maxima[Batch];
        for(size_t q=0;q<Batch;++q) {
            tables[q]=luts[first+q]->lut();
            delta[q]=luts[first+q]->delta();
            offset[q]=luts[first+q]->sum_vl();
            bias[q]=queries[first+q].cb1_sumq;
            maxima[q]=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
        }
        for(size_t b=first_batch;b<=last_batch;++b) {
            size_t begin=std::max(start,b*32)-b*32;
            size_t end=std::min(start+length,b*32+32)-b*32;
            uint32_t mask=uint32_t((uint64_t(1)<<end)-1)&~uint32_t((uint64_t(1)<<begin)-1);
            auto epilogue=[&](size_t q,__m512i a0,__m512i a1) {
                __m512 d,o,c;
                if constexpr(Late) {
                    d=late_broadcast(delta[q]);o=late_broadcast(offset[q]);c=late_broadcast(bias[q]);
                } else {
                    d=_mm512_set1_ps(delta[q]);o=_mm512_set1_ps(offset[q]);c=_mm512_set1_ps(bias[q]);
                }
                __m512 v0=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(d,_mm512_cvtepi32_ps(a0),o),c),
                                        _mm512_loadu_ps(factors+b*32));
                __m512 v1=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(d,_mm512_cvtepi32_ps(a1),o),c),
                                        _mm512_loadu_ps(factors+b*32+16));
                maxima[q]=_mm512_mask_max_ps(maxima[q],__mmask16(mask),maxima[q],v0);
                maxima[q]=_mm512_mask_max_ps(maxima[q],__mmask16(mask>>16),maxima[q],v1);
            };
            if constexpr(Split) accumulate_hacc_queries_split<Batch>(cache.batch_codes(b),tables,dim,epilogue);
            else accumulate_hacc_queries_visit<Batch>(cache.batch_codes(b),tables,dim,epilogue);
        }
        for(size_t q=0;q<Batch;++q)
            score+=_mm512_reduce_max_ps(maxima[q]);
    }
    return score;
}
}
