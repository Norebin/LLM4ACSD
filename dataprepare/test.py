import re
from typing import List, Tuple

'''
    Pre-alpha（Dev）
    Alpha
    Beta
    Perpetual beta
    Open and closed beta
    Release candidate
    Stable release
    Release
'''
def parse_version(version: str) -> Tuple[List[int], int, int]:
    # 如果version有前缀cassandra-的话，将其删除
    # 使用更灵活的正则表达式分割版本号
    if version[0].isalpha():
        match = re.match(r'^[A-Za-z]+(?:-[A-Za-z]+)*-(\d+)(?:[._](\d+))?(?:[._](\d+))?(?:[-._]?([A-Za-z]*(?:[._]*\d*)))?$', version)
    else:
        match = re.match(r'(\d+)(?:[._](\d+))?(?:[._](\d+))?(?:[-._]?([A-Za-z]*(?:[._]*\d*)))?$', version)
    if not match:
        raise ValueError(f"Cannot parse: {version}")

    # 解析主版本号部分
    main_version = [
        int(match.group(1)),  # 主版本号
        int(match.group(2)) if match.group(2) else 0,  # 次版本号
        int(match.group(3)) if match.group(3) and match.group(3).isdigit() else 0  # 补丁版本号
    ]

    # 处理特殊版本标记
    special_version = 0
    special_type = 6

    if match.group(4):
        special_mark = match.group(4)
        if special_mark.startswith(('M', 'm')):
            special_type = 0  # 里程碑版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
        elif special_mark.startswith(('D', 'd')):
            special_type = 1  # Dev版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
        elif special_mark.startswith(('A', 'a')):
            special_type = 2  # Beta版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
        elif special_mark.startswith(('B', 'b')):
            special_type = 3  # Beta版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
        elif special_mark.startswith(('RC', 'rc')):
            special_type = 4  # 候选版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
        elif special_mark == 'final':
            special_type = 5  # 正式版本
            special_version = ''.join(filter(str.isdigit, special_mark)) if special_mark[-1].isdigit() else 0
    return main_version, special_type, special_version


def unique_versions(versions: List[str]) -> List[str]:
    # 排序和去重
    sorted_versions = sorted(versions, key=parse_version)

    unique_result = []
    prev_main_version: Optional[List[int]] = None
    prev_special_type: Optional[int] = None

    for version in sorted_versions:
        curr_main_version, curr_special_type, _ = parse_version(version)
        # 检查是否是同一主版本和同一特殊版本类型
        if (prev_main_version is None or curr_main_version[:2] != prev_main_version[:2] or curr_special_type != prev_special_type):
            unique_result.append(version)
            prev_main_version = curr_main_version
            prev_special_type = curr_special_type

    return unique_result


# 测试版本号排序和去重
# versions = [
#     'cayenne-parent-3.2M1', 'cayenne-parent-3.1.1', 'cayenne-parent-3.1',
#     'cayenne-parent-3.1RC1', 'cayenne-parent-3.1B2', '5.0-M1', '4.2.2',
#     '4.2.1', '4.2', '4.2.RC2', '4.2.RC1', '4.2.M3', '4.2.M2', '4.2.M1',
#     '4.2.B1', '4.1.1', '4.1', '4.1.RC2', '4.1.RC1', '4.1.M2', '4.1.M1',
#     '4.1.B2', '4.1.B1', '4.0.3', '4.0.2', '4.0.1', '4.0', '4.0.RC1',
#     '4.0.M5', '4.0.M4', '4.0.M3', '4.0.M2', '4.0.B2', '4.0.B1', '3.1.3',
#     '3.1.2', '3.1M3', '3.1M2', '3.1M1', '3.1B2', '3.1B1', '3.0.2', '3.0.1',
#     '3.0RC3', '3.0RC2', '3.0RC1', '3.0M6', '3.0M5', '3.0M4', '3.0M3',
#     '3.0M2', '3.0M1', '3.0B1', '3.0-final'
# ]
#
# unique_versions_list = unique_versions(versions)
# print("唯一版本列表：")
# for version in unique_versions_list:
#     print(version)