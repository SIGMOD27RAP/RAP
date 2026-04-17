#include "RaP/dynamic_hnsw.h"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>

using namespace std;

namespace {
vector<vector<float>> rowsFromFlat(const float* p, int32_t n, int32_t d) {
    vector<vector<float>> out(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        const float* row = p + static_cast<size_t>(i) * static_cast<size_t>(d);
        out[static_cast<size_t>(i)].assign(row, row + d);
    }
    return out;
}

vector<int> toIntIds(const vector<size_t>& ids) {
    vector<int> v;
    v.reserve(ids.size());
    for (size_t x : ids) {
        v.push_back(static_cast<int>(x));
    }
    return v;
}

}  // namespace

int main(int argc, char* argv[]) {
    // ---------- 数据与查询 ----------
    int32_t dim = 0;
    int32_t max_elements = 0;
    float* data = nullptr;
    int32_t query_sum = 0;
    int32_t query_dim = 0;
    float* querys = nullptr;

    // ---------- HNSW 与实验参数 ----------
    int circul_sum = 19;
    float delete_ratio = 0.05f;
    string data_file_path = "./dataset/sift/sift_base.fbin";
    string query_file_path = "./dataset/sift/sift_query.fbin";
    string search_result_file_path = "./search_result";
    int K = 10;
    string groundtruth_file_path = "./dataset/sift/gt/gt_"+to_string(K)+"_";
    int init_ef = 32;
    int M = 16;
    int ef = 200;
    int num_threads = 1;

    // ---------- 命令行：可仅用默认路径，亦可传入 11 个参数覆盖 ----------
    if (argc != 12 && argc != 1) {
        cout << "Usage: [M ef K init_ef num_threads circul_sum delete_ratio "
                "data_path query_path gt_prefix result_path]\n";
        return 1;
    }
    if (argc > 1) {
        M = atoi(argv[1]);
        ef = atoi(argv[2]);
        K = atoi(argv[3]);
        init_ef = atoi(argv[4]);
        num_threads = atoi(argv[5]);
        circul_sum = atoi(argv[6]);
        delete_ratio = static_cast<float>(atof(argv[7]));
        data_file_path = argv[8];
        query_file_path = argv[9];
        groundtruth_file_path = argv[10];
        search_result_file_path = argv[11];
    }

    cout << "-----------------------------------------------------------------------------------\n";
    cout << "M: " << M << " ef: " << ef << " K: " << K << " init_ef: " << init_ef
         << " num_threads: " << num_threads
         << " circul_sum: " << circul_sum << " delete_ratio: " << delete_ratio << "\n";
    cout << "data: " << data_file_path << " query: " << query_file_path << "\n";
    cout << "groundtruth_prefix: " << groundtruth_file_path << " result: " << search_result_file_path << "\n";
    cout << "-----------------------------------------------------------------------------------\n";

    // ---------- 读入底库向量（.fbin：n, dim，后接 float 数据） ----------
    cout << "Read data\n";
    ifstream ini_data_reader(data_file_path, ios::binary);
    if (!ini_data_reader.is_open()) {
        cerr << "open data failed: " << data_file_path << "\n";
        return 1;
    }
    ini_data_reader.read(reinterpret_cast<char*>(&max_elements), sizeof(int32_t));
    ini_data_reader.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (max_elements > 1000000000) {
        cout << "cap max_elements at 1000000000\n";
        max_elements = 1000000000;
    }
    data = new float[static_cast<size_t>(dim) * static_cast<size_t>(max_elements)];
    ini_data_reader.read(reinterpret_cast<char*>(data),
                         static_cast<size_t>(max_elements) * static_cast<size_t>(dim) * sizeof(float));
    ini_data_reader.close();

    // ---------- 读入查询 ----------
    readQuerys<float>(query_sum, query_dim, querys, query_file_path);
    if (query_dim != dim) {
        cerr << "query_dim != data dim\n";
        delete[] data;
        delete[] querys;
        return 1;
    }

    cout << "max_elements: " << max_elements << " dim: " << dim << "\n";
    cout << "query_sum: " << query_sum << " query_dim: " << query_dim << "\n";
    cout.flush();
    // ---------- 为 ground truth 准备「按向量一行」的视图（与底库下标一致） ----------
    vector<vector<float>> data_rows = rowsFromFlat(data, max_elements, dim);
    vector<vector<float>> query_rows = rowsFromFlat(querys, query_sum, query_dim);

    // ---------- 初始化 HNSW 索引 ----------
    cout << "Init HNSW\n";
    cout.flush();
    hnswlib::L2Space space(dim);
    auto* alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, static_cast<size_t>(max_elements) + 1u,
                                            static_cast<size_t>(M), static_cast<size_t>(ef), 100u);
    alg_hnsw->setEf(init_ef);
    // ---------- 首批插入：前 index_size 个向量 ----------
    cout << "Add initial points\n";
    cout.flush();
    vector<size_t> initList;
    const int index_size = max_elements / 2;
    for (int i = 0; i < index_size; ++i) {
        initList.push_back(static_cast<size_t>(i));
    }
    double add_latency = addPoint(alg_hnsw, initList, num_threads, data, dim);
    cout << "add_latency (avg ms/point): " << add_latency << "\n";
    cout.flush();

    // ---------- 预生成多轮「删哪些 id / 加哪些 id」（每轮删当前规模的 delete_ratio） ----------
    const int num_rounds = circul_sum;
    vector<vector<size_t>> deleteLists;
    vector<vector<size_t>> addLists;
    deleteLists.reserve(static_cast<size_t>(num_rounds));
    addLists.reserve(static_cast<size_t>(num_rounds));
    for (int r = 0; r < num_rounds; ++r) {
        int delete_start = static_cast<int>(r * static_cast<double>(index_size) * delete_ratio);
        int delete_end =
            static_cast<int>((r + 1) * static_cast<double>(index_size) * delete_ratio);
        int add_start = index_size + delete_start;
        int add_end = index_size + delete_end;
        vector<size_t> d;
        for (int j = delete_start; j < delete_end; ++j) {
            d.push_back(static_cast<size_t>(j));
        }
        vector<size_t> a;
        for (int j = add_start; j < add_end; ++j) {
            a.push_back(static_cast<size_t>(j));
        }
        deleteLists.push_back(move(d));
        addLists.push_back(move(a));
    }

    // ---------- 结果文件 ----------
    ofstream result_writer(search_result_file_path);
    result_writer << "recall,search_ms,del_ms_pt,ins_ms_pt,ef\n";

    // ---------- 首轮：当前索引内点集 = initList，暴力算 GT 后评估 recall ----------
    size_t fileVectorP = 0;
    string gt_path0 = groundtruth_file_path + to_string(fileVectorP) + '-' + to_string(fileVectorP + index_size);
    vector<vector<int>> groundTruthList;
    load_or_calc_groundTruth(gt_path0, query_sum, query_rows, data_rows, toIntIds(initList), groundTruthList, K,
                             false);

    pair<float, double> search_result =
        search_index(alg_hnsw, K, query_sum, query_dim, querys, groundTruthList, num_threads);
    double t1 = search_result.second;
    search_result =
        search_index(alg_hnsw, K, query_sum, query_dim, querys, groundTruthList, num_threads);
    double t2 = search_result.second;
    search_result =
        search_index(alg_hnsw, K, query_sum, query_dim, querys, groundTruthList, num_threads);
    double t3 = search_result.second;
    cout << "initial recall: " << search_result.first << " avg search latency (ms/query): " << (t1 + t2 + t3) / 3 << "\n";
    result_writer << search_result.first << ',' << (t1 + t2 + t3) / 3 << ",,,\n";
    double target_search_latency = (t1 + t2 + t3) / 3;
    vector<int> search_ef_list = {64, 48, 40, 32, 24, 16, 10};
    int cur_ef = std::max(init_ef, K);
    // ---------- 多轮：删点 → 加点 → 更新 GT（仅在新插入集合上精确 topK）→ 再搜索算 recall ----------
    cout << "start rounds\n";
    for (int cir = 0; cir < num_rounds; ++cir) {
        // cout << "round: " << cir << "\n";
        // cout << "deleteLists[cir].size(): " << deleteLists[cir].size() << "\n";
        // cout << "addLists[cir].size(): " << addLists[cir].size() << "\n";
        // cout << "fileVectorP: " << fileVectorP << "\n";
        // cout << "index_size: " << index_size << "\n";
        // cout << "delete range" << deleteLists[cir].front() << " to " << deleteLists[cir].back() << "\n";
        // cout << "add range" << addLists[cir].front() << " to " << addLists[cir].back() << "\n";
        fileVectorP += deleteLists[cir].size();
        double deleteTime;
        deleteTime = deleteIndexRaP(alg_hnsw, deleteLists[cir], num_threads);
        cout << "delete (ms/point): " << deleteTime << "\t";

        double addTime = addPoint(alg_hnsw, addLists[cir], num_threads, data, dim);
        cout << "insert (ms/point): " << addTime << "\t";

        string gt_path =
            groundtruth_file_path + to_string(fileVectorP) + '-' + to_string(fileVectorP + index_size);
        groundTruthList.clear();
        std::vector<size_t> now_list;
        for(size_t i=fileVectorP;i<fileVectorP+index_size;i++){
            now_list.push_back(i);
        }
        load_or_calc_groundTruth(gt_path, query_sum, query_rows, data_rows, toIntIds(now_list), groundTruthList, K, false);
        for(;cur_ef >= K; cur_ef--){
            alg_hnsw->setEf(cur_ef);
            search_result = search_index(alg_hnsw, K, query_sum, query_dim, querys, groundTruthList, num_threads);
            if(search_result.second <= target_search_latency * 1.01 || cur_ef == K){
                cout << "round: " << cir << "\tef: " << cur_ef << "\trecall: " << search_result.first << "\tsearch ms/query: " << search_result.second
                    << "\n";
                cout.flush();
                // 与表头一致：recall, search_ms, del_ms_pt, ins_ms_pt
                result_writer << search_result.first << ',' << search_result.second << ',' << deleteTime << ',' << addTime << ',' << cur_ef
                            << '\n';
                break;
            }
        }
        cur_ef++;
        cur_ef = std::min(cur_ef, init_ef);
        // fileVectorP += deleteLists[cir].size();
    }

    result_writer.close();
    delete alg_hnsw;
    delete[] data;
    delete[] querys;
    return 0;
}
