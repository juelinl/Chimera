#include "ivf_pg.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <omp.h>
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("build_cpu_hnsw CENTROIDS_CPU OUTPUT THREADS");
 std::ifstream in(argv[1],std::ios::binary);uint64_t magic,n,d;
 in.read((char*)&magic,8);in.read((char*)&n,8);in.read((char*)&d,8);
 if(!in||magic!=0x315550434d494843ULL||d!=128||!n)throw std::runtime_error("Invalid centroid file");
 std::vector<float> data(n*d);in.read((char*)data.data(),data.size()*4);if(!in)throw std::runtime_error("Truncated centroids");
 omp_set_num_threads(std::stoi(argv[3]));Chimera::PG_HNSW graph(n,d);
 auto start=std::chrono::steady_clock::now();
 for(size_t begin=0;begin<n;begin+=10000){
  size_t end=std::min(n,begin+10000);
#pragma omp parallel for schedule(static)
  for(size_t i=begin;i<end;++i)graph.hnsw_index->addPoint(data.data()+i*d,i);
  std::cerr<<"centroids "<<end<<"/"<<n<<" elapsed_s "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 }
 std::string tmp=std::string(argv[2])+".tmp";graph.hnsw_index->saveIndex(tmp);std::filesystem::rename(tmp,argv[2]);
 std::cout<<"M=16 efConstruction=500 n="<<n<<" dimension="<<d<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
