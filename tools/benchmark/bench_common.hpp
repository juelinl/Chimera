#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <sys/resource.h>
struct Queries {
 size_t n,l,d;std::vector<float> values;
 explicit Queries(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int shape[3];f.read((char*)shape,12);
  if(!f || shape[0]<=0 || shape[1]<=0 || shape[2]!=128)throw std::runtime_error("Bad query header");
  n=shape[0];l=shape[1];d=shape[2];values.resize(n*l*d);f.read((char*)values.data(),values.size()*4);if(!f)throw std::runtime_error("Truncated queries");
 }
};
inline std::vector<std::unordered_set<size_t>> truth(const std::string& path,size_t n){
 std::ifstream f(path);if(!f)throw std::runtime_error("Missing ground truth");
 std::vector<std::unordered_set<size_t>> t(n);size_t q,d,r;
 while(f>>q>>d>>r){if(q>=n || r<1 || r>100)throw std::runtime_error("Invalid exact-top100 truth");t[q].insert(d);}
 for(auto&s:t)if(s.size()!=100)throw std::runtime_error("Truth must contain 100 distinct exact neighbors");
 return t;
}
inline double pct(std::vector<double> x,double p){std::sort(x.begin(),x.end());return x[size_t(std::ceil(p*x.size()))-1];}
inline void report(std::ostream& out,const std::string& system,int id,int threads,int repeats,
 const std::vector<double>& ms,double recall,size_t queries){
 double total=0;for(auto t:ms)total+=t;struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
 out<<std::setprecision(10)<<"{\"system\":\""<<system<<"\",\"config\":"<<id<<",\"threads\":"<<threads
 <<",\"queries\":"<<queries<<",\"repeats\":"<<repeats<<",\"recall\":"<<recall/ms.size()
 <<",\"qps\":"<<ms.size()*1000/total<<",\"mean_ms\":"<<total/ms.size()<<",\"p50_ms\":"<<pct(ms,.5)
 <<",\"p95_ms\":"<<pct(ms,.95)<<",\"p99_ms\":"<<pct(ms,.99)<<",\"peak_rss_kib\":"<<ru.ru_maxrss<<"}\n"<<std::flush;
}
