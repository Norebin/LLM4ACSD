import pandas as pd

# Function to read the CSV file and filter rows based on class-name containing a specific string
def filter_and_check_classnames(csv_file, filter_string, txt_file):
    # Step 1: 读取CSV文件
    df = pd.read_excel(csv_file)

    # Step 2: 筛选出class-name列包含指定字符串的行
    filtered_df = df[df['class-name'].str.contains(filter_string, na=False)]

    # Step 3: 逐条操作 class-name 列
    with open(txt_file, 'r') as file:
        existing_files = file.read().splitlines()  # 读取所有已存在的文件路径

    # Step 4: 遍历筛选后的 DataFrame
    for _, row in filtered_df.iterrows():
        class_name = row['class-name']
        # 提取 class-name 列中按 split 后的最后一个词
        class_file = class_name.split('.')[-1] + '.java'

        # Step 5: 查找是否存在该 .java 文件
        if class_file not in existing_files:
            print(class_name)  # 如果不存在则输出 class-name


# 示例调用
csv_file = '/model/LiangXJ/developCodeSmell.xlsx'
filter_string = 'mahout'  # 需要筛选的字符串
txt_file = '/model/LiangXJ/CodeSmellProject/apache-mahout-distribution-0.10.2/output.txt'  # 存放已存在的 Java 文件列表的 TXT 文件

filter_and_check_classnames(csv_file, filter_string, txt_file)
