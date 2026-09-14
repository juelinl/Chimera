#pragma once
#include "fastscan_tile.hpp"
namespace Chimera::cpu_kernel {
template<size_t Q,size_t B,class Cache>
inline void score_tile_group(const Cache& cache,size_t dim,
        const std::unique_ptr<rabitqlib::Lut<float>>* luts,const query_object* queries,
        size_t start,size_t length,float& score) {
    const uint8_t* tables[Q];
    __m512 maximum[Q][2],delta[Q],offset[Q],bias[Q];
    for(size_t q=0;q<Q;++q) {
        tables[q]=luts[q]->lut();
        delta[q]=_mm512_set1_ps(luts[q]->delta());
        offset[q]=_mm512_set1_ps(luts[q]->sum_vl());
        bias[q]=_mm512_set1_ps(queries[q].cb1_sumq);
        maximum[q][0]=maximum[q][1]=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
    }
    const float* factors=cache.factors();
    const size_t end_batch=(start+length+31)/32;
    size_t batch=start/32;
    auto update=[&](size_t q,size_t b,__m512i a0,__m512i a1) {
        size_t base=(batch+b)*32;
        size_t lo=std::max(start,base)-base,hi=std::min(start+length,base+32)-base;
        uint32_t mask=uint32_t((uint64_t(1)<<hi)-1)&~uint32_t((uint64_t(1)<<lo)-1);
        __m512 x=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(delta[q],_mm512_cvtepi32_ps(a0),offset[q]),bias[q]),_mm512_loadu_ps(factors+base));
        __m512 y=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(delta[q],_mm512_cvtepi32_ps(a1),offset[q]),bias[q]),_mm512_loadu_ps(factors+base+16));
        maximum[q][0]=_mm512_mask_max_ps(maximum[q][0],__mmask16(mask),maximum[q][0],x);
        maximum[q][1]=_mm512_mask_max_ps(maximum[q][1],__mmask16(mask>>16),maximum[q][1],y);
    };
    for(;batch+B<=end_batch;batch+=B)
        accumulate_hacc_tile<Q,B>(cache.batch_codes(batch),tables,dim,update);
    if constexpr(B>1) for(;batch<end_batch;++batch)
        accumulate_hacc_tile<Q,1>(cache.batch_codes(batch),tables,dim,update);
    for(size_t q=0;q<Q;++q)
        score+=_mm512_reduce_max_ps(_mm512_max_ps(maximum[q][0],maximum[q][1]));
}
template<size_t Q,size_t B,class Cache>
__attribute__((noinline)) float stage2_score_doc_tiled(const Cache& cache,size_t dim,
        const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
        const query_object* queries,size_t nq,size_t start,size_t length) {
    float score=0;
    size_t first=0;
    for(;first+Q<=nq;first+=Q)
        score_tile_group<Q,B>(cache,dim,luts.data()+first,queries+first,start,length,score);
    if constexpr(Q>1) for(;first<nq;++first)
        score_tile_group<1,B>(cache,dim,luts.data()+first,queries+first,start,length,score);
    return score;
}
}
