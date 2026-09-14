#include "../src/cpu_kernel.cpp"
#include <random>
#include "stage2_iteration_kernels.hpp"
#include <cstring>
struct FakeCache {
 std::vector<uint8_t> codes;std::vector<float> scale;
 const uint8_t* batch_codes(size_t b)const{return codes.data()+b*512;}
 const float* factors()const{return scale.data();}
};
int main(){try{
 using namespace Chimera;using namespace Chimera::cpu_kernel;
 std::mt19937 rng(592);std::uniform_real_distribution<float> f(-1,1);
 size_t cases=0;
 for(size_t nq:{size_t(3),size_t(8),size_t(32),size_t(33)}){
  std::vector<float> vectors(nq*128);for(auto&x:vectors)x=f(rng);
  std::vector<query_object> qs(nq);std::vector<std::unique_ptr<rabitqlib::Lut<float>>> luts;
  for(size_t q=0;q<nq;++q){qs[q].cb1_sumq=f(rng);luts.emplace_back(std::make_unique<rabitqlib::Lut<float>>(vectors.data()+q*128,128,true));}
  FakeCache cache;cache.codes.resize(16*512);cache.scale.resize(16*32);
  for(auto&x:cache.codes)x=rng()%256;for(auto&x:cache.scale)x=std::abs(f(rng));
  for(size_t start:{size_t(0),size_t(1),size_t(15),size_t(16),size_t(31),size_t(32)})
  for(size_t length:{size_t(1),size_t(15),size_t(31),size_t(32),size_t(33),size_t(127)}){
   float expected=0;alignas(64) int32_t accum[32];
   for(size_t q=0;q<nq;++q){
    float best=-std::numeric_limits<float>::infinity();
    for(size_t b=start/32;b<=(start+length-1)/32;++b){
     rabitqlib::fastscan::accumulate_hacc(cache.batch_codes(b),luts[q]->lut(),accum,128);
     for(size_t t=std::max(start,b*32);t<std::min(start+length,b*32+32);++t){
      float ip=luts[q]->delta()*float(accum[t%32])+luts[q]->sum_vl();
      best=std::max(best,(ip-qs[q].cb1_sumq)*cache.scale[t]);
     }
    }expected+=best;
   }
   for(float got:{
    stage2_score_doc_compact<1,false,false>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_compact<2,false,false>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_compact<2,true,false>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_experiment<1,true,true>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_experiment<2,false,true>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_experiment<2,true,false>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_experiment<2,true,true>(cache,128,luts,qs.data(),nq,start,length),
    stage2_score_doc_experiment<4,true,true>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_visit<1>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_visit<2>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<1>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<2>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<4>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<8>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<16>(cache,128,luts,qs.data(),nq,start,length),stage2_score_doc_batched<32>(cache,128,luts,qs.data(),nq,start,length)}){
    if(std::memcmp(&got,&expected,sizeof(float)))throw std::runtime_error("MaxSim score mismatch: nq="+std::to_string(nq)+" start="+std::to_string(start)+" length="+std::to_string(length)+" expected="+std::to_string(expected)+" got="+std::to_string(got));
    ++cases;
   }
  }
  // Each fused integer accumulator must exactly match the original.
  const uint8_t* tables[4]={luts[0]->lut(),luts[1]->lut(),luts[2]->lut(),luts[0]->lut()};
  alignas(64) int32_t fused[4][32],single[32];
  accumulate_hacc_queries<4>(cache.batch_codes(0),tables,&fused[0][0],128);
  for(size_t q=0;q<4;++q){
   rabitqlib::fastscan::accumulate_hacc(cache.batch_codes(0),tables[q],single,128);
   if(std::memcmp(single,fused[q],sizeof(single)))throw std::runtime_error("Integer accumulator mismatch");
  }
 }
 // Random integer checks isolate byte-plane recombination from float scoring.
 for(size_t dim:{size_t(64),size_t(128),size_t(256),size_t(512)})
 for(size_t trial=0;trial<64;++trial){
  std::vector<uint8_t> codes(dim*4),storage(dim*8*4);
  for(auto&x:codes)x=rng()%256;for(auto&x:storage)x=rng()%256;
  const uint8_t* tables[4];
  for(size_t q=0;q<4;++q)tables[q]=storage.data()+q*dim*8;
  alignas(64) int32_t got[4][32],expected[32];
  accumulate_hacc_queries_split<4>(codes.data(),tables,dim,[&](size_t q,__m512i a,__m512i b){
   _mm512_storeu_si512(got[q],a);_mm512_storeu_si512(got[q]+16,b);
  });
  for(size_t q=0;q<4;++q){
   rabitqlib::fastscan::accumulate_hacc(codes.data(),tables[q],expected,dim);
   if(std::memcmp(got[q],expected,sizeof(expected)))throw std::runtime_error("Split integer accumulator mismatch");
  }
 }
 std::cout<<"PASS "<<cases<<" bit-identical MaxSim cases; batch1/2/4/8/16/32, query tails, document boundaries; fused/split integer accumulator exact (1024 extra vector checks).\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
