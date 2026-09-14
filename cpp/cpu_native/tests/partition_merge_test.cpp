// Test the restored reducer directly against an independent dense accumulator.
#include "../src/cpu_kernel.cpp"
#include <random>
static_assert(Chimera::cpu_kernel::kStage1DocAccumMode ==
              Chimera::cpu_kernel::stage1_doc_accum_mode::sorted_doc_merge_fast);
static_assert(Chimera::cpu_kernel::kStage3DynamicSchedule);
static_assert(Chimera::cpu_kernel::kDefaultStage2QueryBatch == 1);
int main(){try{
 using namespace Chimera::cpu_kernel;
 std::mt19937 rng(42);partitioned_cross_query_reduce_scratch scratch;
 size_t checked=0;
 for(size_t n: {size_t(1),size_t(7),size_t(1009)})
 for(int scenario=0;scenario<4;++scenario){
  std::vector<std::vector<doc_score_pair>> runs(32);size_t count=0;
  std::vector<float> sums(n,0);std::vector<bool> seen(n,false);
  for(size_t q=0;q<runs.size();++q)for(size_t d=0;d<n;++d){
   bool include=scenario==0?false:scenario==1?(d==n-1):scenario==2?(rng()%17==0):(rng()%2==0);
   if(include){float score=float(rng()%9)/8; runs[q].push_back({int(d),score});sums[d]+=score;seen[d]=true;++count;}
  }
  std::vector<doc_score_pair> expected;
  for(size_t d=0;d<n;++d)if(seen[d])expected.push_back({int(d),sums[d]});
  std::sort(expected.begin(),expected.end(),[](auto a,auto b){return a.score!=b.score?a.score>b.score:a.doc_id<b.doc_id;});
  for(size_t k:{size_t(1),size_t(100),size_t(4000)})for(int threads:{1,8}){
   omp_set_num_threads(threads);std::vector<doc_score_pair> got;bool sorted=false;
   reduce_query_runs_fast_partitioned(runs,count,n,k,scratch,got,sorted);
   if(!sorted||got.size()!=std::min(k,expected.size()))throw std::runtime_error("Wrong reducer result size");
   for(size_t i=0;i<got.size();++i)if(got[i].doc_id!=expected[i].doc_id||got[i].score!=expected[i].score)throw std::runtime_error("Reducer differs from dense oracle");
   ++checked;
  }
 }
 std::cout<<"PASS: "<<checked<<" partition-merge cases; 1/8 threads, ties, empty/skewed/sparse/dense runs, reusable scratch.\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
