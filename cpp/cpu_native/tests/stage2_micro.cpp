// Standalone document-scoring microbenchmark. Includes private templated scorers,
// as the correctness test does; production defaults are not changed.
#include "../src/cpu_kernel.cpp"
#include "quantization.hpp"
#include "stage2_iteration_kernels.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include <random>
#include <fstream>

namespace {
constexpr size_t D=128, NQ=32, QUERY_SETS=8;
// Production dimensions arrive at runtime. Prevent whole-function constant
// specialization from turning this benchmark into a different unrolled kernel.
volatile size_t runtime_dim=D, runtime_nq=NQ;
struct Cache {
 std::vector<uint8_t> codes;std::vector<float> scale;
 const uint8_t* batch_codes(size_t b)const{return codes.data()+b*512;}
 const float* factors()const{return scale.data();}
};
template<class T> void save(std::ofstream&f,const std::vector<T>&v){
 uint64_t n=v.size();f.write((char*)&n,8);f.write((char*)v.data(),v.size()*sizeof(T));
}
template<class T> void read(std::ifstream&f,std::vector<T>&v){
 uint64_t n=0;f.read((char*)&n,8);if(n>uint64_t(1)<<31)throw std::runtime_error("invalid cache size");
 v.resize(n);f.read((char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("truncated cache");
}
}
int main(int argc,char**argv){try{
 using namespace Chimera;using namespace Chimera::cpu_kernel;
 if(argc!=6){std::cerr<<"usage: cpu_stage2_micro CACHE POOL_DOCS PASSES THREADS resident|streaming\n";return 2;}
 const std::string path=argv[1],mode=argv[5];size_t nd=std::stoull(argv[2]),passes=std::stoull(argv[3]);int threads=std::stoi(argv[4]);
 if(nd<64||passes<2||threads<1||(mode!="resident"&&mode!="streaming"))throw std::runtime_error("invalid parameters");
 omp_set_dynamic(0);omp_set_num_threads(threads);
 Cache cache;std::vector<size_t>ptrs;std::vector<float>qvec;
 auto setup=std::chrono::steady_clock::now();
 std::ifstream input(path,std::ios::binary);
 if(input){
  char magic[8];input.read(magic,8);if(std::memcmp(magic,"RQMICRO1",8))throw std::runtime_error("wrong cache format");
  read(input,ptrs);read(input,qvec);read(input,cache.codes);read(input,cache.scale);
  if(ptrs.size()!=nd+1||qvec.size()!=QUERY_SETS*NQ*D||cache.scale.size()%32||
     cache.codes.size()!=cache.scale.size()*16||ptrs.back()>cache.scale.size())throw std::runtime_error("cache configuration mismatch");
 }else{
  std::mt19937 gen(20260914);std::normal_distribution<float>norm(0,1);
  rabitqlib::rotator_impl::FhtKacRotator rotator(D,D,83119);
  alignas(64) float raw[D],rotated[D];
  auto vector=[&](float*out){
   float ss=0;for(size_t k=0;k<D;++k){raw[k]=norm(gen)*(k<16?2.0f:1.0f);ss+=raw[k]*raw[k];}
   for(auto&x:raw)x/=std::sqrt(ss);rotator.rotate(raw,out);
  };
  qvec.resize(QUERY_SETS*NQ*D);for(size_t q=0;q<QUERY_SETS*NQ;++q)vector(qvec.data()+q*D);
  // Varied lengths create nonaligned starts, short documents and long tails.
  const size_t lens[]={1,15,31,32,33,64,96,127,128,129,180,256,384};
  ptrs.push_back(0);for(size_t d=0;d<nd;++d)ptrs.push_back(ptrs.back()+lens[gen()%13]);
  size_t nt=(ptrs.back()+31)/32*32;cache.codes.resize(nt*16);cache.scale.resize(nt);
  alignas(64) uint64_t bits[2];uint8_t unpacked[512];
  for(size_t b=0;b<nt/32;++b){
   for(size_t lane=0;lane<32;++lane){
    vector(rotated);encode_one_bit(rotated,D,bits,&cache.scale[b*32+lane]);
    for(size_t k=0;k<16;++k)unpacked[lane*16+k]=swap_and_reverse_nibbles(((uint8_t*)bits)[k]);
   }
   rabitqlib::fastscan::pack_codes(D,unpacked,32,cache.codes.data()+b*512);
  }
  std::ofstream out(path,std::ios::binary);out.write("RQMICRO1",8);save(out,ptrs);save(out,qvec);save(out,cache.codes);save(out,cache.scale);
  if(!out)throw std::runtime_error("cache write failed");
 }
 std::cerr<<"Prepared "<<nd<<" docs "<<ptrs.back()<<" tokens "<<(cache.codes.size()+cache.scale.size()*4)
 <<" code/factor bytes in "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count()<<" seconds\n";
 struct Queries {std::vector<query_object>q;std::vector<std::unique_ptr<rabitqlib::Lut<float>>>luts;};
 std::vector<Queries>queries(QUERY_SETS);
 for(size_t h=0;h<QUERY_SETS;++h)for(size_t q=0;q<NQ;++q){
  float*v=qvec.data()+(h*NQ+q)*D;queries[h].q.emplace_back(v,D,4);queries[h].luts.emplace_back(std::make_unique<rabitqlib::Lut<float>>(v,D,true));
 }
 auto score=[&](int variant,size_t doc,size_t qi){
  auto&qs=queries[qi];size_t start=ptrs[doc],len=ptrs[doc+1]-start;
  const size_t dim=runtime_dim,nq=runtime_nq;
  if(variant==0)return stage2_score_doc_batched<1>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==1)return stage2_score_doc_visit<1>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==2)return stage2_score_doc_batched<2>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==3)return stage2_score_doc_visit<2>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==5)return stage2_score_doc_experiment<2,false,true>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==6)return stage2_score_doc_experiment<2,true,false>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==7)return stage2_score_doc_experiment<2,true,true>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==8)return stage2_score_doc_compact<2,false,false>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  if(variant==9)return stage2_score_doc_compact<2,true,false>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
  return stage2_score_doc_batched<32>(cache,dim,qs.luts,qs.q.data(),nq,start,len);
 };
 // Validate actual packed code/LUT mapping independently of the optimized scorer:
 // scalar original FastScan supplies an oracle for all representative lengths.
 for(size_t doc=0;doc<std::min(nd,size_t(128));++doc)for(size_t qi=0;qi<QUERY_SETS;++qi){
  float oracle=0;auto&qs=queries[qi];alignas(64) int32_t accum[32];
  for(size_t q=0;q<NQ;++q){float best=-std::numeric_limits<float>::infinity();
   for(size_t b=ptrs[doc]/32;b<=(ptrs[doc+1]-1)/32;++b){
    rabitqlib::fastscan::accumulate_hacc(cache.batch_codes(b),qs.luts[q]->lut(),accum,D);
    for(size_t t=std::max(ptrs[doc],b*32);t<std::min(ptrs[doc+1],b*32+32);++t){
     float ip=qs.luts[q]->delta()*float(accum[t%32])+qs.luts[q]->sum_vl();
     best=std::max(best,(ip-qs.q[q].cb1_sumq)*cache.scale[t]);
    }
   }oracle+=best;
  }
  for(int v=0;v<10;++v){float got=score(v,doc,qi);if(std::memcmp(&got,&oracle,sizeof(float)))throw std::runtime_error("oracle bit mismatch");}
 }
 std::vector<size_t>order(nd);std::iota(order.begin(),order.end(),0);
 std::mt19937 shuffle_rng(9391);std::shuffle(order.begin(),order.end(),shuffle_rng);
 const size_t work=mode=="resident"?4000:nd,active=mode=="resident"?64:nd;
 std::vector<float>out(work),expected(work);
 std::vector<size_t>docs(work);
 const char*names[]={"baseline_b1","visit_b1","baseline_b2","visit_b2","baseline_b32","late_b2","split_b2","split_late_b2","compact_b2","compact_split_b2"};

 std::vector<int> selected;
 if(const char* wanted=std::getenv("CHIMERA_MICRO_VARIANTS")) {
  std::istringstream input(wanted);std::string name;
  while(input>>name){
   int v=0;while(v<10&&name!=names[v])++v;
   if(v==10)throw std::runtime_error("Unknown timing variant: "+name);
   selected.push_back(v);
  }
  if(selected.empty())throw std::runtime_error("Empty timing variant selection");
 } else {for(int v=0;v<10;++v)selected.push_back(v);}
 std::cout<<std::setprecision(10);
 // Two warmup passes; rotate variant order to distribute thermal/order effects.
 for(int pass=-2;pass<int(passes);++pass){
  size_t qi=size_t(pass+2)%QUERY_SETS,tokens=0;
  for(size_t i=0;i<work;++i){docs[i]=order[(i+size_t(pass+2)*work)%active];tokens+=ptrs[docs[i]+1]-ptrs[docs[i]];}
  // The fixed candidate list and query are identical for every variant in this pass.
  for(size_t i=0;i<work;++i)expected[i]=score(0,docs[i],qi);
  for(size_t j=0;j<selected.size();++j){
   int v=selected[(j+pass+2)%selected.size()];
   // Amortize frequency/team ramp-up: each variant runs for roughly
   // 0.2 seconds or more, rather than drawing conclusions from one 5-ms call.
   const size_t inner=mode=="resident"?(threads==1?8:48):(threads==1?1:6);
   auto begin=std::chrono::steady_clock::now();
   for(size_t iteration=0;iteration<inner;++iteration){
#pragma omp parallel for schedule(dynamic,1)
    for(size_t i=0;i<work;++i)out[i]=score(v,docs[i],qi);
   }
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/inner;
   double checksum=0;for(size_t i=0;i<work;++i){if(std::memcmp(&out[i],&expected[i],sizeof(float)))throw std::runtime_error("timed score mismatch");checksum+=out[i];}
   if(pass>=0)std::cout<<"{\"variant\":\""<<names[v]<<"\",\"mode\":\""<<mode<<"\",\"pass\":"<<pass
    <<",\"inner_repeats\":"<<inner<<",\"threads\":"<<threads<<",\"docs\":"<<work<<",\"tokens\":"<<tokens<<",\"ms\":"<<ms
    <<",\"ns_per_pair\":"<<ms*1e6/(tokens*NQ)<<",\"checksum\":"<<checksum<<"}\n"<<std::flush;
  }
 }
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
