import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 加载模型和Tokenizer ---
model_name = "/model/LiangXJ/Model/Qwen/Qwen2.5-Coder-14B-Instruct"
print(f"开始加载模型: {model_name}")

try:
    # 尝试使用 bfloat16 以节省显存，如果你的 GPU 不支持，可以换成 torch.float16
    # device_map="auto" 会自动将模型分片加载到可用的 GPU 和 CPU 上
    # trust_remote_code=True 是因为 Qwen 模型结构定义在 Hugging Face Hub 上
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # 或者 torch.bfloat16, torch.float16
        device_map="auto",  # "auto" 需要 accelerate 库
        trust_remote_code=True
    )
    print(f"模型 {model_name} 加载成功!")
    print(f"模型当前运行设备: {model.device}") # 对于 device_map="auto"，会显示第一个参数的设备

except Exception as e:
    print(f"加载模型时发生错误: {e}")
    print("请确保已安装 `transformers`, `torch`, `accelerate` 并且有足够的显存/内存。")
    print("对于 Qwen 模型，可能还需要 `einops` 和 `transformers_stream_generator`。")
    exit()

# --- 2. 输出模型结构 ---
print("\n--- 模型结构 ---")
print(model)

# 获取模型的配置信息
config = model.config
num_hidden_layers = config.num_hidden_layers
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
vocab_size = config.vocab_size

print(f"\n--- 模型配置摘要 ---")
print(f"词汇表大小 (Vocab Size): {vocab_size}")
print(f"隐藏层大小 (Hidden Size): {hidden_size}")
print(f"Transformer层数 (Number of Hidden Layers): {num_hidden_layers}")
print(f"注意力头数 (Number of Attention Heads): {num_attention_heads}")
print(f"激活函数 (Hidden Activation): {config.hidden_act}")
print(f"中间层大小 (Intermediate Size in FFN): {config.intermediate_size}")

# 详细查看每一层的类型 (Qwen2 模型通常是 Qwen2DecoderLayer)
# Qwen2 模型的 Transformer blocks 通常在 model.model.layers
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    print(f"\n--- Transformer Decoder 层列表 (共 {len(model.model.layers)} 层) ---")
    for i, layer in enumerate(model.model.layers):
        print(f"Layer {i}: {type(layer)}")
        # 你可以进一步打印层内的子模块，例如：
        # print(f"  - Self Attention: {type(layer.self_attn)}")
        # print(f"  - MLP: {type(layer.mlp)}")
else:
    print("未能定位到 Transformer 层列表，模型结构可能与预期不同。")


# --- 3. 生成时取出指定中间层的 hidden state ---
print("\n--- 生成文本并提取 Hidden States ---")

# 准备输入
# prompt = "def fibonacci(n):"
prompt = "你好，请介绍一下你自己" # 使用中文 prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")

# 将输入移动到模型所在的设备 (如果模型是分片的，移动到第一个参数所在的设备)
# 对于 device_map="auto"，模型可能分布在多个设备上，
# inputs 需要在 model.device (通常是第一个分片所在的设备)
# 如果模型在 "meta" device (表示未完全加载到具体设备)，则需要先移动到具体设备
# 但由于我们用了 device_map="auto"，模型已经加载到具体设备上了
if model.device.type != 'meta':
    model_inputs = model_inputs.to(model.device)
else:
    print("警告: 模型在 'meta' device，输入可能无法正确处理。")


# 生成文本，并要求输出 hidden states
# `output_hidden_states=True` 会在输出中包含所有层的 hidden states
# `return_dict_in_generate=True` 使输出更易于处理
# `max_new_tokens` 控制生成token的数量，设置小一点以加快速度和减少内存占用
with torch.no_grad(): # 在推理时不需要计算梯度
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=10, # 生成少量 token 作为演示
        output_hidden_states=True,
        return_dict_in_generate=True
    )

# `outputs.hidden_states` 的结构:
# 这是一个元组，长度等于生成的 token 数量 (不包括 prompt 中的 token)。
# 每个元素对应一个生成步骤（即生成一个新 token 的步骤）。
# 每个元素本身又是一个元组，包含了该生成步骤中所有层的 hidden states：
#   - 第0个元素: embedding 层的输出
#   - 第1个元素: 第1个 Transformer Decoder 层的输出
#   - ...
#   - 第 N 个元素: 第 N 个 Transformer Decoder 层的输出 (N = num_hidden_layers)
#   - 第 N+1 个元素 (如果存在): 最后一个 LayerNorm 层的输出 (取决于模型结构)
#
# 每个 hidden state 张量的形状: (batch_size, sequence_length_at_this_step, hidden_size)
# 注意：sequence_length_at_this_step 会随着 token 的生成而增加。

generated_sequence = outputs.sequences[0]
generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print(f"\n生成的文本: {generated_text}")

if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
    print(f"\n--- 提取 Hidden States ---")
    num_generated_tokens = len(outputs.hidden_states)
    print(f"共为 {num_generated_tokens} 个新生成的 token 提取了 hidden states。")

    # 假设我们想获取第1个新生成的token的hidden states
    token_step_index = 0 # 0 表示第一个新生成的token
    if num_generated_tokens > token_step_index:
        hidden_states_for_first_new_token = outputs.hidden_states[token_step_index]
        # hidden_states_for_first_new_token 是一个元组，包含 (embeddings, layer_1_hs, ..., layer_L_hs)
        # 长度通常是 num_hidden_layers + 1 (因为包含了 embedding 层的输出)
        print(f"第一个新生成 token 的 hidden_states 元组长度: {len(hidden_states_for_first_new_token)}")
        print(f"其中包含 {len(hidden_states_for_first_new_token)-1} 个 Transformer 层的 hidden states (加上 embedding 层)。")

        # 假设我们想获取第 5 层 (0-indexed) 的 hidden state
        # 注意：hidden_states_for_first_new_token[0] 是 embedding
        # hidden_states_for_first_new_token[k+1] 对应第 k 个 Transformer 层的输出
        target_layer_index = 5 # 想要第 5 层 (0-indexed)
        if len(hidden_states_for_first_new_token) > target_layer_index + 1:
            # +1 是因为第0个是 embedding
            hs_layer_5_first_token = hidden_states_for_first_new_token[target_layer_index + 1]
            # hs_layer_5_first_token 的形状: (batch_size, sequence_length_at_this_step, hidden_size)
            # sequence_length_at_this_step = len(prompt_tokens) + token_step_index + 1
            print(f"为第一个新生成的 token，提取第 {target_layer_index} 层 (0-indexed) 的 hidden state。")
            print(f"  形状: {hs_layer_5_first_token.shape}") # (batch_size, seq_len_at_that_step, hidden_dim)

            # 通常我们关心的是用于预测下一个token的那个位置的hidden state，即序列最后一个token的hidden state
            last_token_hs_layer_5 = hs_layer_5_first_token[:, -1, :] # (batch_size, hidden_dim)
            print(f"  该层最后一个 token 的 hidden state 形状: {last_token_hs_layer_5.shape}")
            print(f"  该 hidden state 的一部分: {last_token_hs_layer_5[0, :5]}...") # 打印前5个维度
        else:
            print(f"错误: 目标层索引 {target_layer_index} 超出范围。可用的 Transformer 层数: {len(hidden_states_for_first_new_token)-1}")

        # 假设我们想获取最后一层 (num_hidden_layers - 1) 的 hidden state
        last_layer_index = num_hidden_layers - 1
        hs_last_layer_first_token = hidden_states_for_first_new_token[last_layer_index + 1]
        print(f"为第一个新生成的 token，提取最后一层 (第 {last_layer_index} 层) 的 hidden state。")
        print(f"  形状: {hs_last_layer_first_token.shape}")
        last_token_hs_last_layer = hs_last_layer_first_token[:, -1, :]
        print(f"  该层最后一个 token 的 hidden state 形状: {last_token_hs_last_layer.shape}")

    else:
        print("没有生成新的 token，无法提取 hidden states。")

    # 如果你想获取所有生成步骤中，特定层的 hidden state (通常是最后一个 token 的)
    target_layer_to_collect = 10 # 假设我们对第10层感兴趣 (0-indexed)
    collected_hs_for_layer_10 = []
    if num_hidden_layers > target_layer_to_collect:
        for token_idx in range(num_generated_tokens):
            # outputs.hidden_states[token_idx] 是一个元组 (embeddings, layer_0_hs, ..., layer_L-1_hs)
            # 我们需要第 target_layer_to_collect + 1 个元素
            hs_at_this_step_for_target_layer = outputs.hidden_states[token_idx][target_layer_to_collect + 1]
            # hs_at_this_step_for_target_layer 的形状是 (batch_size, current_sequence_length, hidden_size)
            # 我们取最后一个 token 的 hidden state
            last_token_hs = hs_at_this_step_for_target_layer[:, -1, :] # Shape: (batch_size, hidden_size)
            collected_hs_for_layer_10.append(last_token_hs)

        if collected_hs_for_layer_10:
            # 将列表中的张量堆叠起来
            # 结果形状: (batch_size, num_generated_tokens, hidden_size)
            collected_hs_for_layer_10_tensor = torch.stack(collected_hs_for_layer_10, dim=1)
            print(f"\n为所有 {num_generated_tokens} 个新生成的 token，收集的第 {target_layer_to_collect} 层 (0-indexed) 的最后一个位置的 hidden states:")
            print(f"  形状: {collected_hs_for_layer_10_tensor.shape}")
            print(f"  第一个生成token的该层hidden state (部分): {collected_hs_for_layer_10_tensor[0, 0, :5]}...")
    else:
        print(f"目标收集层 {target_layer_to_collect} 超出模型层数 {num_hidden_layers}")

else:
    print("输出中未找到 'hidden_states'。请确保 `output_hidden_states=True` 已设置。")

# --- 也可以通过单次 forward pass 获取 hidden states (不用于生成，而是对固定输入获取表示) ---
print("\n--- 单次 Forward Pass 提取 Hidden States (非生成模式) ---")
# 使用原始的 model_inputs
with torch.no_grad():
    forward_outputs = model(**model_inputs, output_hidden_states=True)

# `forward_outputs.hidden_states` 的结构:
# 这是一个元组，长度为 num_hidden_layers + 1 (embedding + L layers)
# 每个元素是对应层的输出 hidden state 张量
# 张量形状: (batch_size, sequence_length, hidden_size)
# sequence_length 是输入序列的长度

if hasattr(forward_outputs, "hidden_states") and forward_outputs.hidden_states is not None:
    print(f"Forward pass 的 hidden_states 元组长度: {len(forward_outputs.hidden_states)}")
    # 第0个是 embedding output
    embedding_output = forward_outputs.hidden_states[0]
    print(f"Embedding output 形状: {embedding_output.shape}")

    # 获取第5层 (0-indexed) 的 hidden state
    target_layer_index_forward = 5
    if len(forward_outputs.hidden_states) > target_layer_index_forward + 1:
        hs_layer_5_forward = forward_outputs.hidden_states[target_layer_index_forward + 1]
        print(f"第 {target_layer_index_forward} 层 (0-indexed) 的 hidden state 形状 (forward pass): {hs_layer_5_forward.shape}")
        # 对于 forward pass，通常我们可能对所有 token 的 hidden states 都感兴趣，或者对 [CLS] token (如果有)，或者对最后一个 token
        last_token_hs_layer_5_forward = hs_layer_5_forward[:, -1, :]
        print(f"  该层最后一个 token 的 hidden state 形状: {last_token_hs_layer_5_forward.shape}")
        print(f"  该 hidden state 的一部分: {last_token_hs_layer_5_forward[0, :5]}...")
    else:
        print(f"错误: 目标层索引 {target_layer_index_forward} 超出范围。")
else:
    print("Forward pass 输出中未找到 'hidden_states'。")