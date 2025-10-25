# 代码气味可操作性分类 - 特征融合模型

本项目实现了一个融合代码语义特征和代码结构特征的分类系统，用于判断代码气味的可操作性。

## 功能概述

本系统主要实现以下功能：
1. 加载并处理两种特征：
   - 从PT文件加载大模型提取的代码语义特征
   - 从GraphML文件构建并提取图结构特征
2. 使用多种注意力机制实现特征融合
3. 基于融合特征进行二元分类（可操作/不可操作）
4. 评估模型性能并输出多种评估指标

## 目录结构

```
/model/lxj/LLM4ACS/RQ2/
├── fusion_model.py          # 主要模型实现
├── compare_fusion_methods.py # 比较不同融合方法的工具
├── compare_all_projects.py   # 比较所有项目结果的工具
├── run_fusion_experiments.sh # 单个项目的实验脚本
├── run_all_projects.sh       # 批量处理多个项目的脚本
└── README.md                # 本说明文件
```

## 支持的融合方法

系统实现了以下几种特征融合方法：
1. `self_attention`: 自注意力融合
2. `cross_attention`: 交叉注意力融合
3. `gate`: 门控融合机制
4. `multihead`: 多头注意力融合

## 使用方法

### 1. 运行单个项目的所有融合方法实验

```bash
bash run_fusion_experiments.sh <project_name>
```
例如：
```bash
bash run_fusion_experiments.sh cassandra
```

### 2. 批量运行多个项目的实验

```bash
bash run_all_projects.sh
```

### 3. 比较单个项目中不同融合方法的效果

```bash
python compare_fusion_methods.py --project <project_name> --results_dir ./results
```
例如：
```bash
python compare_fusion_methods.py --project cassandra --results_dir ./results
```

### 4. 比较所有项目的综合结果

```bash
python compare_all_projects.py --results_dir ./results
```

## 实验结果

实验结果将保存在以下目录中：

- 单个融合方法的结果: `./results/<project>_<fusion_type>_results.csv`
- 单个项目的比较结果: `./results/comparison/<project>_comparison.csv`
- 所有项目的综合结果: `./results/overall/`

图表结果将以PNG格式保存在相应的结果目录中。

## 参数设置

主要可调参数包括：

- `--semantic_dim`: 语义特征维度 (默认: 1024)
- `--graph_dim`: 图特征维度 (默认: 128)
- `--fusion_dim`: 融合特征维度 (默认: 256)
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批量大小 (默认: 32)
- `--lr`: 学习率 (默认: 0.001)

## 依赖项

- Python 3.6+
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- networkx

## 数据路径

- 语义特征文件: `/model/lxj/LLM_comment_generate/temp/semantic_features/`
- CSV标签文件: `/model/lxj/actionableSmell/`
- 图文件: `/model/data/R-SE/tools/graphgen/`

## 注意事项

1. 确保所有数据路径正确且数据格式符合预期
2. 系统使用CUDA加速，请确保GPU资源可用
3. 建议在运行全部项目实验前，先测试单个项目的运行效果
