#include "cpu_kernel.hpp"
#include "io.hpp"
#ifdef CHIMERA_VTUNE
#include <ittnotify.h>
#endif
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "bench_common.hpp"
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("cpu_bench INDEX QUERY TRUTH WARMUP CONFIG THREADS REPEAT OUT");
 int threads=std::stoi(argv[6]),repeats=std::stoi(argv[7]);
 if(threads<1||repeats<1)throw std::runtime_error("Invalid thread/repeat count");
 omp_set_dynamic(0);omp_set_num_threads(threads);
 Chimera::cpu_mvr_index index(argv[1],Chimera::cpu_quantized_payload_mode::PreferDoc4ExSidecar);
 index.set_doc_mapping(Chimera::load_doclens(std::string(argv[1])+"/doclens.bin"));
 Chimera::cpu_kernel::clustered_stage1_cache stage1(argv[1],index);
 Chimera::cpu_kernel::packed_token_stream_cache stage2(index);
 Queries queries(argv[2]),warm(argv[4]);auto gt=truth(argv[3],queries.n);
 if(queries.l!=32||warm.l!=32)throw std::runtime_error("This benchmark requires 32-token queries for the established partitioned path");
 std::ifstream conf(argv[5]);std::ofstream out(argv[8]),profiles(std::string(argv[8])+".profiles.jsonl");
 if(!conf||!out||!profiles)throw std::runtime_error("Invalid file arguments");
 int id,route_ef;Chimera::gpu_search_runtime_options o;
 bool validate=std::getenv("CHIMERA_VALIDATE_THREADS")!=nullptr;
 while(conf>>id>>o.nprobe>>o.k_rank_cluster>>o.k_rank_all_tokens>>route_ef){
  if(route_ef!=0||o.nprobe<1||size_t(o.nprobe)>index.n_clusters||o.k_rank_all_tokens<100||o.k_rank_cluster<o.k_rank_all_tokens)throw std::runtime_error("Invalid native config; routing field must be 0 (native HNSW ef policy)");
  const bool compare=std::getenv("CHIMERA_CPU_STAGE2_COMPARE")!=nullptr;
  const char* batch_env=std::getenv("CHIMERA_CPU_STAGE2_BATCH");
  std::vector<int> batches=compare?std::vector<int>{0,1,2,4,0}:std::vector<int>{batch_env?std::stoi(batch_env):Chimera::cpu_kernel::kDefaultStage2QueryBatch};
  if(compare && std::getenv("CHIMERA_CPU_STAGE2_COMPARE_BATCHES")){
   batches.clear();std::istringstream list(std::getenv("CHIMERA_CPU_STAGE2_COMPARE_BATCHES"));int b;
   while(list>>b)batches.push_back(b);
   if(batches.empty())throw std::runtime_error("Empty comparison batch list");
  }
  const char* kernel_env=std::getenv("CHIMERA_CPU_STAGE2_KERNEL");
  std::vector<std::string> kernels(batches.size(),kernel_env?kernel_env:"baseline");
  if(compare&&std::getenv("CHIMERA_CPU_STAGE2_COMPARE_KERNELS")){
   kernels.clear();std::istringstream list(std::getenv("CHIMERA_CPU_STAGE2_COMPARE_KERNELS"));std::string k;
   while(list>>k)kernels.push_back(k);
   if(kernels.size()!=batches.size())throw std::runtime_error("Kernel and batch comparison lists must match");
  }
  for(auto&k:kernels)if(k!="baseline"&&k!="visit")throw std::runtime_error("Unknown comparison kernel");
  std::vector<std::vector<size_t>> baseline_ids(queries.n);
  size_t variant_index=0;
  for(int batch:batches){
  const auto& kernel=kernels[variant_index];setenv("CHIMERA_CPU_STAGE2_KERNEL",kernel.c_str(),1);
  std::string batch_string=std::to_string(batch);setenv("CHIMERA_CPU_STAGE2_BATCH",batch_string.c_str(),1);
  std::cerr<<"Begin stage2 batch="<<batch<<" config="<<id<<"\n";
  for(size_t q=0;q<warm.n;++q){
   const float* v=&warm.values[q*warm.l*warm.d];
   if(validate){
    omp_set_num_threads(1);auto a=Chimera::cpu_kernel::search(index,stage1,stage2,v,warm.l,100,o);
    omp_set_num_threads(threads);auto b=Chimera::cpu_kernel::search(index,stage1,stage2,v,warm.l,100,o);
    Chimera::cpu_kernel::search_profile p;auto c=Chimera::cpu_kernel::search_profiled(index,stage1,stage2,v,warm.l,100,o,&p,false);
    if(a!=b||b!=c)throw std::runtime_error("Rank mismatch between serial/parallel/profiled paths at warmup query "+std::to_string(q));
   }else Chimera::cpu_kernel::search(index,stage1,stage2,v,warm.l,100,o);
  }
  if(validate)std::cerr<<"Validation passed: "<<warm.n<<" queries, identical top100 at 1/"<<threads<<" threads and profiled/unprofiled.\n";
  std::vector<double> times;double recall=0;Chimera::cpu_kernel::search_profile sum;
#ifdef CHIMERA_VTUNE
  __itt_resume();
#endif
  for(int r=0;r<repeats;++r)for(size_t q=0;q<queries.n;++q){
   Chimera::cpu_kernel::search_profile p;auto start=std::chrono::steady_clock::now();
   auto hits=Chimera::cpu_kernel::search_profiled(index,stage1,stage2,&queries.values[q*queries.l*queries.d],queries.l,100,o,&p,false);
   times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
   if(hits.size()!=100||std::unordered_set<size_t>(hits.begin(),hits.end()).size()!=100)throw std::runtime_error("Invalid top100 results");
   if(compare){
    if(variant_index==0 && r==0)baseline_ids[q]=hits;
    else if(hits!=baseline_ids[q])throw std::runtime_error("Stage2 variant changed top100 at query "+std::to_string(q));
   }
   size_t found=0;for(auto h:hits)found+=gt[q].count(h);recall+=found/100.0;
   Chimera::cpu_kernel::accumulate_search_profile(sum,p);
  }
#ifdef CHIMERA_VTUNE
  __itt_pause();
#endif
  report(out,compare?("Chimera-CPU-native-batch"+batch_string+(kernel=="visit"?"-visit":"")):"Chimera-CPU-native",id,threads,repeats,times,recall,queries.n);
  auto p=Chimera::cpu_kernel::average_search_profile(sum,times.size());
  profiles<<std::setprecision(12)<<"{\"config\":"<<id<<",\"threads\":"<<threads<<",\"routing\":\"HNSW\",\"merge\":\"sorted_doc_merge_fast_partitioned\",\"hnsw_ef\":"<<Chimera::PG_HNSW::search_ef_for_k(o.nprobe);
#define FIELD(name) profiles<<",\"" #name "\":"<<p.name
  FIELD(query_setup_ms);FIELD(stage1_cluster_ms);FIELD(stage1_probe_ms);FIELD(stage1_prepare_ms);FIELD(stage1_scan_ms);FIELD(stage1_reduce_ms);FIELD(stage1_reduce_merge_ms);FIELD(stage1_reduce_cross_query_merge_ms);FIELD(stage1_reduce_sort_ms);FIELD(stage1_reduce_serial_setup_ms);FIELD(stage1_reduce_partition_accumulate_work_ms);FIELD(stage1_reduce_partition_sort_work_ms);FIELD(stage1_cleanup_ms);
  FIELD(stage2_1bit_ms);FIELD(stage2_lut_ms);FIELD(stage2_score_docs_ms);FIELD(stage2_select_topk_ms);FIELD(stage2_materialize_ms);
  FIELD(stage3_exbits_ms);FIELD(stage3_prepare_ms);FIELD(stage3_score_docs_ms);FIELD(stage3_select_topk_ms);FIELD(total_ms);FIELD(end_to_end_ms);
#undef FIELD
  profiles<<",\"stage2_batch\":"<<batch<<",\"stage2_kernel\":\""<<kernel<<"\"}\n"<<std::flush;
  std::cerr<<"Completed stage2 batch="<<batch<<"; rank checks passed\n";
  ++variant_index;
  }
 }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
