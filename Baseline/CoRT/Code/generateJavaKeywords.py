def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        # 跳过空行
        if not line.strip():
            continue
            
        # # 跳过以#开头的行
        if line.strip().startswith('#'):
            continue
            
        # 对于其他行,只保留第一个空格之前的内容
        parts = line.split(' ', 1)
        if parts:
            processed_lines.append(parts[0])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

# 使用示例
input_file = '/model/lxj/CoRT/Code/keywords.txt'  # 输入文件路径
output_file = 'Java-keyword.txt'  # 输出文件路径
process_file(output_file, output_file)