#pragma once
#include "fastscan_query_split.hpp"
namespace Chimera::cpu_kernel {
// Half a packed block: use only one nibble per code byte and four accumulators/query.
template<size_t Q,bool High,class Visitor>
inline void accumulate_hacc_half(const uint8_t* codes,const uint8_t* const* tables,size_t dim,Visitor&& visit) {
    __m512i acc[Q][2][2];
    for(auto& q:acc)for(auto& p:q)for(auto& a:p)a=_mm512_setzero_si512();
    const __m512i mask=_mm512_set1_epi8(15);
    for(size_t m=0;m<dim/4;m+=4) {
        __m512i c=_mm512_loadu_si512(codes+m*16);
        if constexpr(High)c=_mm512_srli_epi16(c,4);
        c=_mm512_and_si512(c,mask);
        for(size_t q=0;q<Q;++q)for(size_t p=0;p<2;++p) {
            __m512i x=_mm512_shuffle_epi8(_mm512_loadu_si512(tables[q]+m/4*128+p*64),c);
            acc[q][p][0]=_mm512_add_epi16(acc[q][p][0],x);
            acc[q][p][1]=_mm512_add_epi16(acc[q][p][1],_mm512_srli_epi16(x,8));
        }
    }
    for(size_t q=0;q<Q;++q)
        visit(q,_mm512_add_epi32(reduce_byte_accumulators(acc[q][0][0],acc[q][0][1]),
                  _mm512_slli_epi32(reduce_byte_accumulators(acc[q][1][0],acc[q][1][1]),8)));
}
template<size_t Q,class Cache>
inline void score_half_group(const Cache& cache,size_t dim,
        const std::unique_ptr<rabitqlib::Lut<float>>* luts,const query_object* queries,
        size_t start,size_t length,float& score) {
    const uint8_t* tables[Q];__m512 maximum[Q],delta[Q],offset[Q],bias[Q];
    for(size_t q=0;q<Q;++q){
        tables[q]=luts[q]->lut();maximum[q]=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
        delta[q]=_mm512_set1_ps(luts[q]->delta());offset[q]=_mm512_set1_ps(luts[q]->sum_vl());bias[q]=_mm512_set1_ps(queries[q].cb1_sumq);
    }
    for(size_t h=start/16;h<(start+length+15)/16;++h){
        size_t base=h*16,lo=std::max(start,base)-base,hi=std::min(start+length,base+16)-base;
        __mmask16 mask=__mmask16(((uint32_t(1)<<hi)-1)&~((uint32_t(1)<<lo)-1));
        auto update=[&](size_t q,__m512i a) {
            __m512 v=_mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(delta[q],_mm512_cvtepi32_ps(a),offset[q]),bias[q]),_mm512_loadu_ps(cache.factors()+base));
            maximum[q]=_mm512_mask_max_ps(maximum[q],mask,maximum[q],v);
        };
        if(h&1)accumulate_hacc_half<Q,true>(cache.batch_codes(h/2),tables,dim,update);
        else accumulate_hacc_half<Q,false>(cache.batch_codes(h/2),tables,dim,update);
    }
    for(size_t q=0;q<Q;++q)score+=_mm512_reduce_max_ps(maximum[q]);
}
template<size_t Q,class Cache>
__attribute__((noinline)) float stage2_score_doc_half(const Cache& cache,size_t dim,
        const std::vector<std::unique_ptr<rabitqlib::Lut<float>>>& luts,
        const query_object* queries,size_t nq,size_t start,size_t length) {
    float score=0;size_t first=0;
    for(;first+Q<=nq;first+=Q)score_half_group<Q>(cache,dim,luts.data()+first,queries+first,start,length,score);
    if constexpr(Q>1)for(;first<nq;++first)score_half_group<1>(cache,dim,luts.data()+first,queries+first,start,length,score);
    return score;
}
}
