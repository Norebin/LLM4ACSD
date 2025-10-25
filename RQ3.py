import os
import subprocess

# 1. 模型和数据集列表
# 模型路径列表
model_paths = [
    "/model/LiangXJ/Model/microsoft/phi-4",
    "/model/LiangXJ/Model/deepseek-ai/deepseek-coder-6.7b-instruct",
    "/model/LiangXJ/Model/CodeLlama-13b-Instruct-hf",
    "/model/LiangXJ/Model/Qwen/Qwen2.5-Coder-14B-Instruct",
    "/model/LiangXJ/Model/PEFT/codellama_13B/lora_all_merge",
    "/model/LiangXJ/Model/PEFT/codellama_13B/dora_all_merge",
    "/model/LiangXJ/Model/PEFT/deepseek-coder6.7-instruct/dora_all_merge",
    "/model/LiangXJ/Model/PEFT/deepseek-coder6.7-instruct/lora_all_merge",
    "/model/LiangXJ/Model/PEFT/phi4_14B/dora_all_merge",
    "/model/LiangXJ/Model/PEFT/phi4_14B/lora_all_merge",
    "/model/LiangXJ/Model/PEFT/Qwen2.5-Coder-14B-Instruct/dora_all_merge",
    "/model/LiangXJ/Model/PEFT/Qwen2.5-Coder-14B-Instruct/lora_all_merge",
]

# 数据集列表
datasets = [
    "test_cassandra",
    "test_cayenne",
    "test_dbeaver",
    "test_dubbo",
    "test_easyexcel",
    "test_jedis",
    "test_pig",
    "test_stirling",
    "test_struts",
    "test_mockito",
    "test_jsoup"
]

# 2. 辅助函数
def get_template(model_path: str) -> str:
    """根据模型路径确定使用的template"""
    model_name_lower = model_path.lower()
    if "codellama" in model_name_lower:
        return "default"
    elif "deepseek-coder" in model_name_lower:
        return "deepseekcoder"
    elif "phi-4" in model_name_lower or "phi4" in model_name_lower:
        return "phi4"
    elif "qwen" in model_name_lower:
        return "qwen"
    # 默认值，以防万一
    return "default"

def get_model_short_name(model_path: str) -> str:
    """从模型路径中提取一个简短且唯一的名称"""
    if "PEFT" in model_path:
        # 对于PEFT模型，使用最后两级目录名组合
        parts = model_path.split('/')
        return f"{parts[-2]}_{parts[-1]}"
    else:
        # 对于基础模型，使用最后一级目录名
        return model_path.split('/')[-1]

# 3. 主逻辑
def main():
    """主函数，用于执行所有测试"""
    # 结果保存目录
    save_dir = "/model/lxj/LLM4ACS/RQ3/"
    os.makedirs(save_dir, exist_ok=True)

    # 推理脚本路径
    infer_script = "/model/lxj/LLaMA-Factory/scripts/vllm_infer.py"

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"

    for model_path in model_paths:
        template = get_template(model_path)
        model_short_name = get_model_short_name(model_path)
        
        print(f"--- 开始测试模型: {model_short_name} ---")

        # 任务列表：首先是单个数据集，然后是所有数据集的组合
        tasks = datasets + [",".join(datasets)]
        
        for dataset_name in tasks:
            if dataset_name == ",".join(datasets):
                save_file_name = f"{model_short_name}_all_datasets.jsonl"
                current_dataset_log_name = "all_datasets"
            else:
                save_file_name = f"{model_short_name}_{dataset_name}.jsonl"
                current_dataset_log_name = dataset_name

            save_path = os.path.join(save_dir, save_file_name)
            if os.path.exists(save_path):
                print(f"[跳过] 文件已存在: {save_path}")
                continue
            command = [
                "python", infer_script,
                "--model_name_or_path", model_path,
                "--template", template,
                "--dataset", dataset_name,
                "--save_name", save_path
            ]
            
            print(f"\n[执行命令] 模型: {model_short_name}, 数据集: {current_dataset_log_name}")
            print(f"命令: {' '.join(command)}")

            try:
                subprocess.run(command, check=True, env=env)
                print(f"[成功] 模型: {model_short_name}, 数据集: {current_dataset_log_name} 的测试已完成。")
            except subprocess.CalledProcessError as e:
                print(f"[错误] 模型: {model_short_name}, 数据集: {current_dataset_log_name} 的测试失败: {e}")
            except FileNotFoundError:
                print(f"[错误] 无法找到脚本: {infer_script}。请检查路径是否正确。")
                return # 如果脚本不存在，则停止执行

        print(f"--- 模型: {model_short_name} 的所有测试已完成 ---\n")

    print("所有模型的测试均已执行完毕。")

if __name__ == "__main__":
    # 脚本执行前的重要提示
    print("重要提示：请确保您已经激活了 'lxj_llamaFac' conda 环境。")
    print("脚本将在3秒后开始执行...")
    # time.sleep(3) # 如果需要，可以取消注释以获得准备时间
    main()