#!/bin/bash

# 运行所有融合方法的比较实验
# 使用方法: ./run_fusion_experiments.sh <project_name>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <project_name>"
    echo "Example: $0 cassandra"
    exit 1
fi

PROJECT=$1
OUTPUT_DIR="/model/lxj/LLM4ACS/RQ2/results1"

# 创建结果目录
mkdir -p $OUTPUT_DIR

# 定义融合方法
FUSION="progress"
# Model_Names=("origin_phi4" "lora_phi4" "dora_phi4" "origin_codellama" "lora_codellama" "dora_codellama" "origin_deepseek" "lora_deepseek" "dora_deepseek" "origin_qwen" "lora_qwen" "dora_qwen")
Model_Names=("lora_phi4" "lora_codellama" "lora_deepseek" "lora_qwen")
# 对每种融合方法运行实验
for model_name in "${Model_Names[@]}"; do
    echo "===================================="
    echo "Running experiment with model: $model_name"
    echo "===================================="
    
    python /model/lxj/LLM4ACS/RQ2/fusion_model.py \
        --project $PROJECT \
        --model_name $model_name \
        --output_dir $OUTPUT_DIR \
        --fusion_type $FUSION \
        --epochs 200 \
        --batch_size 256 \
        --lr 0.0005
        
    echo "Experiment with $model_name completed."
    echo
done
# --fusion_type $FUSION \
# # 结果比较
# echo "===================================="
# echo "Comparing results of different fusion methods"
# echo "===================================="

# python /model/lxj/LLM4ACS/RQ2/compare_fusion_methods.py \
#     --project $PROJECT \
#     --results_dir $OUTPUT_DIR

# echo "All experiments completed."