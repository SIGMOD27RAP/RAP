#!/usr/bin/env bash
set -euo pipefail

M=16
ef=200
K=10
init_ef=32
num_threads=160
circul_sum=19
delete_rate=0.05
datasets=(mnist)

make clean
make

for dataset_type in "${datasets[@]}"; do
  dataset_path="./dataset/${dataset_type}/${dataset_type}_base.fbin"
  query_set_path="./dataset/${dataset_type}/${dataset_type}_query.fbin"
  groundtruth_path="./dataset/${dataset_type}/gt/gt_${K}_"
  search_result_path="./result/${dataset_type}/hnsw_${dataset_type}_M${M}_ef${ef}_thr${num_threads}_Drate${delete_rate}_K${K}_init_ef${init_ef}.csv"

  mkdir -p "./result/${dataset_type}"

  echo "============================================================"
  echo "Running dataset: ${dataset_type}"
  echo "data: ${dataset_path}"
  echo "query: ${query_set_path}"
  echo "gt: ${groundtruth_path}"
  echo "result: ${search_result_path}"
  echo "============================================================"

  ./dynamic_test \
    "${M}" "${ef}" "${K}" "${init_ef}" "${num_threads}" "${circul_sum}" "${delete_rate}" \
    "${dataset_path}" "${query_set_path}" "${groundtruth_path}" "${search_result_path}"
done

echo "All datasets finished."
