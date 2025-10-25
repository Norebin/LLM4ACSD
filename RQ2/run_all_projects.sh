#!/bin/bash

# 批量处理多个项目的实验
# 使用方法: ./run_all_projects.sh

OUTPUT_DIR="/model/lxj/LLM4ACS/RQ2/results1"

# 创建结果目录
mkdir -p $OUTPUT_DIR

# 定义要处理的项目
PROJECTS=(
    "cassandra" 
    "cayenne" 
    "dbeaver" 
    "dubbo" 
    "easyexcel" 
    "jedis" 
    "jsoup" 
    "mockito" 
    "pig" 
    "struts"
)

# 对每个项目运行实验
for project in "${PROJECTS[@]}"; do
    echo "========================================================"
    echo "Processing project: $project"
    echo "========================================================"
    
    # 使用之前创建的脚本运行所有融合方法
    bash /model/lxj/LLM4ACS/RQ2/run_fusion_experiments.sh $project
    
    echo "Project $project completed."
    echo
done

# # 创建所有项目的汇总比较
# echo "========================================================"
# echo "Creating overall comparison for all projects"
# echo "========================================================"

# python /model/lxj/LLM4ACS/RQ2/compare_all_projects.py \
#     --results_dir $OUTPUT_DIR

# echo "All projects processed successfully."
