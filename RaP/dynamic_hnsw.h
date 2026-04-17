#include "hnswlib.h"
#include "bruteforce.h"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <queue>
#include <chrono>
#include <unistd.h>
#include <stdio.h>
#include <random>
#include <omp.h>

using namespace std;
typedef unsigned int tableint;
namespace fs = std::filesystem;

inline double elapsed_ms(chrono::steady_clock::time_point t0, chrono::steady_clock::time_point t1) {
    return chrono::duration<double, milli>(t1 - t0).count();
}

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

template <typename dist_t>
static inline dist_t l2_sqr(const dist_t* a, const dist_t* b, size_t dim) {
    dist_t s = (dist_t)0;
    for (size_t i = 0; i < dim; i++) {
        const dist_t d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// groundTruth_path 不存在，或 recalc=true：并行暴力精确计算每个 query 的 topK（在 data_id_list 指定的点集合里），按距离从小到大写入 groundTruth，并保存到文件。
// groundTruth_path 存在，且 recalc=false：从文件读取 groundTruth。
// 文件格式（二进制）：int32 query_sum, int32 K, 然后 query_sum*K 个 int32 id。
template <typename dist_t>
inline void load_or_calc_groundTruth(
    const std::string& groundTruth_path,
    int32_t query_sum,
    const std::vector<std::vector<dist_t>>& querys,
    const std::vector<std::vector<dist_t>>& data,
    const std::vector<int>& data_id_list,
    std::vector<std::vector<int>>& groundTruth,
    int32_t K,
    bool recalc) {
    if (!recalc && fs::exists(groundTruth_path)) {
        std::ifstream in(groundTruth_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("open groundTruth_path for read failed: " + groundTruth_path);
        }
        int32_t file_q = 0;
        int32_t file_k = 0;
        in.read((char*)&file_q, sizeof(int32_t));
        in.read((char*)&file_k, sizeof(int32_t));
        if (!in.good()) {
            throw std::runtime_error("read groundTruth header failed: " + groundTruth_path);
        }
        if (file_q != query_sum || file_k != K) {
            throw std::runtime_error("groundTruth header mismatch (query_sum/K): " + groundTruth_path);
        }
        groundTruth.assign(query_sum, std::vector<int>(K, -1));
        for (int32_t i = 0; i < query_sum; i++) {
            in.read((char*)groundTruth[i].data(), sizeof(int32_t) * (size_t)K);
        }
        if (!in.good()) {
            throw std::runtime_error("read groundTruth body failed: " + groundTruth_path);
        }
        return;
    }

    if ((int32_t)querys.size() != query_sum) {
        throw std::runtime_error("query_sum != querys.size()");
    }
    if (query_sum > 0 && data_id_list.empty()) {
        groundTruth.assign(query_sum, std::vector<int>(K, -1));
        return;
    }

    const size_t dim = query_sum > 0 ? querys[0].size() : 0;
    for (int32_t i = 0; i < query_sum; i++) {
        if (querys[i].size() != dim) {
            throw std::runtime_error("inconsistent query dim");
        }
    }
    for (int id : data_id_list) {
        if (id < 0 || (size_t)id >= data.size()) {
            throw std::runtime_error("data_id_list contains out-of-range id");
        }
        if (data[(size_t)id].size() != dim) {
            throw std::runtime_error("data/query dim mismatch");
        }
    }

    groundTruth.assign(query_sum, std::vector<int>(K, -1));
    const int32_t K_eff = std::min<int32_t>(K, (int32_t)data_id_list.size());
    const size_t num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());

    ParallelFor(0, (size_t)query_sum, num_threads, [&](size_t qi, size_t /*threadId*/) {
        std::vector<std::pair<dist_t, int>> best;
        best.reserve(data_id_list.size());
        const dist_t* qptr = querys[qi].data();
        for (int id : data_id_list) {
            const dist_t d = l2_sqr(qptr, data[(size_t)id].data(), dim);
            best.emplace_back(d, id);
        }
        if (K_eff > 0) {
            std::nth_element(best.begin(), best.begin() + (K_eff - 1), best.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            best.resize((size_t)K_eff);
            std::sort(best.begin(), best.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (int32_t k = 0; k < K_eff; k++) {
                groundTruth[(size_t)qi][(size_t)k] = best[(size_t)k].second;
            }
        }
    });

    fs::path p(groundTruth_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    std::ofstream out(groundTruth_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("open groundTruth_path for write failed: " + groundTruth_path);
    }
    out.write((const char*)&query_sum, sizeof(int32_t));
    out.write((const char*)&K, sizeof(int32_t));
    for (int32_t i = 0; i < query_sum; i++) {
        out.write((const char*)groundTruth[i].data(), sizeof(int32_t) * (size_t)K);
    }
    if (!out.good()) {
        throw std::runtime_error("write groundTruth failed: " + groundTruth_path);
    }
}
template <typename dist_t>
void readInitData(int32_t &dim,int32_t &max_elements,dist_t*& data,string data_file_path){
    // read data
    cout<<"Read data"<<endl;
    std::ifstream data_reader(data_file_path, std::ios::binary);
    if(!data_reader.is_open()){
        cout<<"data_reader open failed!!!!"<<endl;
        throw;
    }
    data_reader.read((char*)&max_elements,sizeof(int32_t));
    data_reader.read((char*)&dim,sizeof(int32_t));
    if(max_elements>10000000){
        cout<<"max_elements>10000000"<<endl;
        max_elements=10000000;
    }
    data = new dist_t[dim * max_elements];
    data_reader.read((char*)data, max_elements * dim * sizeof(dist_t));
    data_reader.close();
}

template <typename dist_t>
void readQuerys(int32_t &query_sum,int32_t &query_dim,dist_t*& querys,string query_file_path){
    // Read query
    cout<<"Read query"<<endl;
    std::ifstream query_reader(query_file_path, std::ios::binary);
    if(!query_reader.is_open()){
        cout<<"query_reader open failed!!!!"<<endl;
        throw;
    }
    query_reader.read((char*)&query_sum,sizeof(int32_t));
    query_reader.read((char*)&query_dim,sizeof(int32_t));
    querys = new dist_t[query_dim * query_sum];
    query_reader.read((char*)querys, query_sum * query_dim * sizeof(dist_t));
    query_reader.close();
}

template <typename dist_t>
void readGroundTruth(int32_t &groundtruth_sum,int32_t &groundtruth_dim,uint32_t*& groundtruth,string groundtruth_file_path,int K){
    // Read groundtruth
    // cout<<"Read groundtruth"<<endl;
    std::ifstream groundtruth_reader(groundtruth_file_path, std::ios::binary);
    if(!groundtruth_reader.is_open()){
        cout<<"groundtruth_reader open failed!!!!"<<endl;
        throw;
    }
    groundtruth_reader.read((char*)&groundtruth_sum,sizeof(int32_t));
    groundtruth_reader.read((char*)&groundtruth_dim,sizeof(int32_t));
    if(groundtruth==nullptr){
        groundtruth = new uint32_t[groundtruth_dim * groundtruth_sum];
    }
    groundtruth_reader.read((char*)groundtruth, groundtruth_sum * groundtruth_dim * sizeof(uint32_t));
    groundtruth_reader.close();
    if(groundtruth_dim<K){
        cout<<"groundtruth_dim too small!!!!"<<endl;
        throw;
    }
}

template <typename dist_t>
void creat_index(hnswlib::HierarchicalNSW<dist_t>*& alg_hnsw,
    hnswlib::L2Space *space,
    int M,
    int ef,
    int32_t dim,
    int32_t max_elements){
    alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, max_elements+1, M, ef, 100);
}

template <typename dist_t>
pair<float,double> search_index(
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,
    int K,
    int32_t query_sum,
    int32_t query_dim,
    dist_t* querys,
    std::vector<std::vector<int>> groundTruthList,
    int num_threads){
    float correct_sum = 0;
    float* corrects = new float[query_sum];
    memset(corrects,0,query_sum*sizeof(float));
    double search_time_sum = 0;
    double* search_times = new double[query_sum];
    memset(search_times,0,query_sum*sizeof(double));
    vector<std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>> result(query_sum);
    
    const auto t_search0 = chrono::steady_clock::now();
    ParallelFor(0, query_sum, 1, [&](size_t row, size_t threadId) {
        // std::cout << "row: " << row << std::endl;
        // std::cout.flush();
        result[row] = alg_hnsw->searchKnn((void*)((char*)querys + row * query_dim * sizeof(dist_t)), K);
    });
    const auto t_search1 = chrono::steady_clock::now();
    search_time_sum = elapsed_ms(t_search0, t_search1);
    
    ParallelFor(0, query_sum, num_threads, [&](size_t row, size_t threadId) {
        while (!result[row].empty()) {
            const hnswlib::labeltype label = result[row].top().second;
            auto it = std::find(groundTruthList[row].begin(), groundTruthList[row].end(),
                                static_cast<int>(label));
            if (it != groundTruthList[row].end()) {
                corrects[row] += 1;
            }
            result[row].pop();
        }
    });

    for(int i=0;i<query_sum;i++){
        correct_sum+=corrects[i];
    }

    float recall = correct_sum / query_sum / K;
    // 单次查询平均耗时（毫秒）
    return make_pair(recall, search_time_sum / query_sum);
}

void creat_deleteList(
    vector<size_t>&deleteList,
    int pre_creat_sum,
    int list_len,
    default_random_engine& random,
    uniform_int_distribution<size_t>& dis1){

    deleteList.clear();
    vector<size_t> deleteList_temp;
    while(deleteList.size()<list_len){
        for(int i=0;i<pre_creat_sum;i++){
            deleteList_temp.emplace_back((size_t)(dis1(random)));
        }
        sort(deleteList_temp.begin(),deleteList_temp.end());
        for(int i=0;i<deleteList_temp.size()-1;i++){
            if(deleteList_temp[i]!=deleteList_temp[i+1]){
                deleteList.emplace_back(deleteList_temp[i]);
                if(deleteList.size()>=list_len){
                    break;
                }
            }
        }
        pre_creat_sum*=2;
    }
    random_shuffle(deleteList.begin(),deleteList.end());
}

void write_Vector(vector<size_t> vec,string file_path){
    std::ofstream vec_writer(file_path);
    for(int i=0;i<vec.size();i++){
        vec_writer<<vec[i];
        if(i!=vec.size()-1)
            vec_writer<<"\n";
    }
    vec_writer.close();
}


template <typename dist_t>
double deleteIndexRaP(
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,
    vector<size_t> deleteList,
    int num_threads){
    const auto t0 = chrono::steady_clock::now(); 
    // for(int row=0;row<deleteList.size();row++){
    //     alg_hnsw->deleteNeighMark(deleteList[row]);
    // }
    ParallelFor(0, deleteList.size(), num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->deleteNeighMark(deleteList[row]);
    });
    const auto t1 = chrono::steady_clock::now();
    const double delete_time = elapsed_ms(t0, t1);
    return delete_time / deleteList.size();
}

template <typename dist_t>
double addPoint(
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw,
    vector<size_t> addList,
    int num_threads,
    dist_t* data,
    int32_t dim){
    double addTime_avg=0;
    vector<double> addTimes(addList.size(),0);
    const auto t0 = chrono::steady_clock::now();

    ParallelFor(0, addList.size(), num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)((char*)data + dim * addList[row] * sizeof(dist_t)), addList[row]);
    });
    const auto t1 = chrono::steady_clock::now();
    addTime_avg = elapsed_ms(t0, t1);
    return addTime_avg / addList.size();
}

void show_progress_bar(int progress, int total) {
    const int bar_width = 100;
    float progress_percentage = (float)progress / total;
    int pos = bar_width * progress_percentage;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << " ";
    }

    std::cout << "] " << int(progress_percentage * 100.0) << "%";
    std::flush(std::cout);
}

// 对于每个 query，暴力精确计算 topK（在 data_offset..data_offset+data_len-1 内），按距离从小到大写入 groundtruth（每行 K 个）。
template <typename dist_t>
inline void calculate_groundtruth(
    int32_t query_sum,
    int32_t query_dim,
    dist_t* querys,
    uint32_t* groundtruth,
    int32_t data_offset,
    int32_t data_len,
    dist_t* data,
    int32_t dim,
    int K,
    int num_threads) {
    hnswlib::L2Space space(dim);
    auto* alg_bf = new hnswlib::BruteforceSearch<dist_t>(space, (size_t)data_len);
    for (int32_t i = 0; i < data_len; i++) {
        alg_bf->addPoint((void*)((char*)data + (size_t)(data_offset + i) * (size_t)dim * sizeof(dist_t)), (size_t)(data_offset + i));
    }
    ParallelFor(0, (size_t)query_sum, (size_t)num_threads, [&](size_t row, size_t /*threadId*/) {
        auto pq = alg_bf->searchKnn((void*)((char*)querys + row * (size_t)query_dim * sizeof(dist_t)), K);
        std::vector<uint32_t> tmp;
        tmp.reserve((size_t)K);
        while (!pq.empty()) {
            tmp.push_back((uint32_t)pq.top().second);
            pq.pop();
        }
        std::reverse(tmp.begin(), tmp.end()); // 从小到大
        const size_t base = row * (size_t)K;
        for (int k = 0; k < K; k++) {
            groundtruth[base + (size_t)k] = (k < (int)tmp.size()) ? tmp[(size_t)k] : (uint32_t)0;
        }
    });
    delete alg_bf;
}