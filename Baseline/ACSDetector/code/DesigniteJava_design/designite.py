

# s10 = 106+63+195+115+131+185
# s000 = 207+205+735+294+639+451
# s111 = 257+205+736+295+639+451

# s10 = 653+610+543+81+242+903
# s000 = 980+3034+1785+346+789+2543
# s111 = 980+3034+1785+346+789+2543

s10 = 903
s000 = 2543
s111 = 2543



true_positives = s10  # 真正例
false_positives = s111  # 假正例
false_negatives = 0  # 假负例
true_negatives = s10+s000  # 真负例

print('TP, FP, FN, TN: ', true_positives, false_positives, false_negatives, true_negatives)
# 计算精确度（Precision）
precision = true_positives / (true_positives + false_positives)
# 计算召回率（Recall）
recall = true_positives / (true_positives + false_negatives)
# 计算 F1 分数
f1 = 2 * (precision * recall) / (precision + recall)
# 计算虚警率（False Positive Rate）
false_positive_rate = false_positives / (false_positives + true_negatives)

p = float(format(precision, '.4f'))
r = float(format(recall, '.4f'))
f1 = float(format(f1, '.4f'))
fpr = float(format(false_positive_rate, '.4f'))
print('p, r, f1, fpr, auc: ', p, r, f1, fpr)