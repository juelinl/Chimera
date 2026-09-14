// Synthetic 2D tile exploration. No production dispatch is changed.
#include "../src/cpu_kernel.cpp"
#include "stage2_tile_kernels.hpp"
#include "stage2_half_tile_kernels.hpp"
#include <fstream>
#include <random>
#include <sstream>
using namespace Chimera;
using namespace Chimera::cpu_kernel;
namespace {
struct Cache {
    size_t dim=128;
    std::vector<uint8_t> codes;std::vector<float> scale;
    const uint8_t* batch_codes(size_t b)const{return codes.data()+b*dim*4;}
    const float* factors()const{return scale.data();}
};
using Luts=std::vector<std::unique_ptr<rabitqlib::Lut<float>>>;
using Fn=float(*)(const Cache&,size_t,const Luts&,const query_object*,size_t,size_t,size_t);
struct Variant {std::string name;size_t q,b;Fn fn;};
std::vector<Variant> variants() {
    std::vector<Variant> v;
    v.push_back({"baseline_b1",1,1,stage2_score_doc_batched<1,Cache>});
    v.push_back({"visit_b2",2,1,stage2_score_doc_visit<2,Cache>});
#define TILE(Q,B) v.push_back({"q" #Q "_d" #B,Q,B,stage2_score_doc_tiled<Q,B,Cache>});
#define ROW(Q) TILE(Q,1) TILE(Q,2) TILE(Q,4)
    ROW(1) ROW(2) ROW(4) ROW(8) ROW(16) ROW(32)
#undef ROW
#undef TILE
    v.push_back({"baseline_b8",8,1,stage2_score_doc_batched<8,Cache>});
    v.push_back({"baseline_b32",32,1,stage2_score_doc_batched<32,Cache>});
#define HALF(Q) v.push_back({"q" #Q "_h16",Q,0,stage2_score_doc_half<Q,Cache>});
    HALF(1) HALF(2) HALF(4) HALF(8) HALF(16) HALF(32)
#undef HALF
    v.push_back({"q3_d1",3,1,stage2_score_doc_tiled<3,1,Cache>});
    v.push_back({"q6_d1",6,1,stage2_score_doc_tiled<6,1,Cache>});
    v.push_back({"q3_h16",3,0,stage2_score_doc_half<3,Cache>});
    v.push_back({"q6_h16",6,0,stage2_score_doc_half<6,Cache>});
    return v;
}
float reference(const Cache& c,size_t dim,const Luts& l,const query_object* q,size_t nq,size_t start,size_t len) {
    alignas(64) int32_t a[32];float score=0;
    for(size_t j=0;j<nq;++j){
        float best=-std::numeric_limits<float>::infinity();
        for(size_t b=start/32;b<=(start+len-1)/32;++b){
            rabitqlib::fastscan::accumulate_hacc(c.batch_codes(b),l[j]->lut(),a,dim);
            for(size_t t=std::max(start,b*32);t<std::min(start+len,b*32+32);++t)
                best=std::max(best,(l[j]->delta()*float(a[t%32])+l[j]->sum_vl()-q[j].cb1_sumq)*c.scale[t]);
        }
        score+=best;
    }
    return score;
}
template<class T>void read(std::ifstream& f,std::vector<T>& v){
    uint64_t n=0;f.read((char*)&n,8);
    if(n>(uint64_t(1)<<31))throw std::runtime_error("Invalid fixture size");
    v.resize(n);f.read((char*)v.data(),n*sizeof(T));
    if(!f)throw std::runtime_error("Truncated fixture");
}
struct Query {std::vector<query_object> q;Luts luts;};
void setup_query(Query& q,float* data,size_t nq,size_t dim) {
    for(size_t j=0;j<nq;++j){
        q.q.emplace_back(data+j*dim,dim,4);
        q.luts.emplace_back(std::make_unique<rabitqlib::Lut<float>>(data+j*dim,dim,true));
    }
}
void selftest(const std::vector<Variant>& vs){
    std::mt19937 rng(51793);std::uniform_real_distribution<float> dist(-1,1);
    size_t checks=0;
    for(size_t dim:{size_t(64),size_t(128),size_t(256)})
    for(size_t nq:{size_t(3),size_t(32),size_t(33),size_t(64),size_t(65),size_t(128)}){
        Cache c;c.dim=dim;c.codes.resize(512*dim/8);c.scale.resize(512);
        for(auto& x:c.codes)x=rng()%256;
        for(auto& x:c.scale)x=0.5f+std::abs(dist(rng));
        std::vector<float> data(nq*dim);for(auto& x:data)x=dist(rng);
        Query query;setup_query(query,data.data(),nq,dim);
        for(size_t start:{size_t(0),size_t(1),size_t(15),size_t(31),size_t(32)})
        for(size_t len:{size_t(1),size_t(31),size_t(32),size_t(65),size_t(257)}){
            float ref=reference(c,dim,query.luts,query.q.data(),nq,start,len);
            for(const auto& v:vs){
                float got=v.fn(c,dim,query.luts,query.q.data(),nq,start,len);
                if(std::memcmp(&got,&ref,sizeof(float)))throw std::runtime_error("Tile correctness: "+v.name+" nq="+std::to_string(nq)+" dim="+std::to_string(dim));
                ++checks;
            }
        }
    }
    std::cout<<"PASS "<<checks<<" exact float-bit comparisons across 28 tiles and 4 controls, dimensions 64/128/256, nq 3/32/33/64/65/128, document boundaries and tails\n";
}
}
int main(int argc,char**argv){try{
    auto all=variants();
    if(argc==2&&std::string(argv[1])=="--self-test"){selftest(all);return 0;}
    if(argc!=8)throw std::runtime_error("usage: cpu_stage2_tiles FIXTURE NQ varied|512 THREADS resident|streaming PASSES ORDER_OFFSET");
    size_t nq=std::stoull(argv[2]),passes=std::stoull(argv[6]),order_offset=std::stoull(argv[7]);
    int threads=std::stoi(argv[4]);std::string layout=argv[3],mode=argv[5];
    if((nq!=32&&nq!=64&&nq!=128)||threads<1||passes<2||(layout!="varied"&&layout!="512")||(mode!="resident"&&mode!="streaming"))throw std::runtime_error("Invalid configuration");
    omp_set_dynamic(0);omp_set_num_threads(threads);
    Cache cache;std::vector<size_t>ptrs;std::vector<float>qvec;
    std::ifstream f(argv[1],std::ios::binary);char magic[8]={};f.read(magic,8);
    if(std::memcmp(magic,"RQMICRO1",8))throw std::runtime_error("Invalid fixture");
    read(f,ptrs);read(f,qvec);read(f,cache.codes);read(f,cache.scale);
    if(qvec.size()!=256*128||ptrs.empty()||ptrs.back()>cache.scale.size()||cache.codes.size()!=cache.scale.size()*16)throw std::runtime_error("Fixture shape mismatch");
    if(layout=="512"){
        size_t nt=ptrs.back();ptrs.clear();for(size_t t=0;t<nt;t+=512)ptrs.push_back(t);ptrs.push_back(nt);
    }
    const size_t nd=ptrs.size()-1,sets=256/nq;
    std::vector<Query>queries(sets);
    for(size_t s=0;s<sets;++s)setup_query(queries[s],qvec.data()+s*nq*128,nq,128);
    std::vector<Variant> selected;
    if(const char* filter=std::getenv("CHIMERA_TILE_VARIANTS")){
        std::istringstream in(filter);std::string name;
        while(in>>name){
            auto it=std::find_if(all.begin(),all.end(),[&](const Variant& v){return v.name==name;});
            if(it==all.end())throw std::runtime_error("Unknown tile: "+name);
            selected.push_back(*it);
        }
    } else selected=all;
    if(selected.empty())throw std::runtime_error("Empty tile selection");
    std::vector<size_t>order(nd);std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(9391);std::shuffle(order.begin(),order.end(),rng);
    size_t work=mode=="resident"?512:nd,active=mode=="resident"?64:nd;
    std::vector<size_t>docs(work);
    size_t tokens=0;
    for(size_t i=0;i<work;++i){docs[i]=order[i%active];tokens+=ptrs[docs[i]+1]-ptrs[docs[i]];}
    std::vector<std::vector<float>>expected(sets,std::vector<float>(work));
    for(size_t s=0;s<sets;++s){
#pragma omp parallel for schedule(dynamic,1)
        for(size_t i=0;i<work;++i)expected[s][i]=all[0].fn(cache,128,queries[s].luts,queries[s].q.data(),nq,ptrs[docs[i]],ptrs[docs[i]+1]-ptrs[docs[i]]);
        // Independent scalar reduction oracle on 32 selected docs, every tile.
        for(size_t i=0;i<32;++i){
            size_t d=docs[i];float ref=reference(cache,128,queries[s].luts,queries[s].q.data(),nq,ptrs[d],ptrs[d+1]-ptrs[d]);
            for(const auto& v:all){
                float got=v.fn(cache,128,queries[s].luts,queries[s].q.data(),nq,ptrs[d],ptrs[d+1]-ptrs[d]);
                if(std::memcmp(&got,&ref,4))throw std::runtime_error("Fixture oracle mismatch: "+v.name);
            }
        }
    }
    std::vector<float>out(work);
    // Fixed inner repetition count per condition, independent of tile performance.
    size_t inner=mode=="resident"?std::max(size_t(1),size_t(threads)*16*32/nq/(layout=="512"?4:1)):1;
    std::cout<<std::setprecision(10);
    volatile size_t runtime_dim=128,runtime_nq=nq;
    for(int pass=-1;pass<int(passes);++pass){
        size_t s=size_t(pass+1)%sets;
        for(size_t j=0;j<selected.size();++j){
            const auto& v=selected[(j+pass+1+order_offset)%selected.size()];
            const size_t dim=runtime_dim,nquery=runtime_nq;
            auto begin=std::chrono::steady_clock::now();
            for(size_t repeat=0;repeat<inner;++repeat){
#pragma omp parallel for schedule(dynamic,1)
                for(size_t i=0;i<work;++i){size_t d=docs[i];out[i]=v.fn(cache,dim,queries[s].luts,queries[s].q.data(),nquery,ptrs[d],ptrs[d+1]-ptrs[d]);}
            }
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/inner;
            double checksum=0;
            for(size_t i=0;i<work;++i){if(std::memcmp(&out[i],&expected[s][i],4))throw std::runtime_error("Timed score mismatch: "+v.name);checksum+=out[i];}
            if(pass>=0)std::cout<<"{\"variant\":\""<<v.name<<"\",\"query_tile\":"<<v.q<<",\"document_tile\":"<<(v.b?v.b*32:16)<<",\"nq\":"<<nq<<",\"layout\":\""<<layout<<"\",\"mode\":\""<<mode<<"\",\"threads\":"<<threads<<",\"pass\":"<<pass<<",\"ms\":"<<ms<<",\"docs\":"<<work<<",\"tokens\":"<<tokens<<",\"inner_repeats\":"<<inner<<",\"ns_per_pair\":"<<ms*1e6/(tokens*nq)<<",\"checksum\":"<<checksum<<"}\n"<<std::flush;
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
