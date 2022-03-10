import numpy as np
from scipy import stats
# 使用avg 衡量加权方法
amount = 8000
s_log = np.load('results/2201_s.npy')
s_log = s_log[:amount]
s_sum = np.load('results/2201_s_sum.npy')
s_sum = s_sum[:amount]
s_sum = s_sum[~np.isinf(s_log)]
s_auc = np.load('results/2201_s_auv.npy')
s_auc = s_auc[:amount]
s_auc = s_auc[~np.isinf(s_log)]
accs = np.load('results/2201_acc.npy')
accs = accs[:amount]
accs = accs[~np.isinf(s_log)]

s_log = s_log[~np.isinf(s_log)]

s_log_avg = s_log / np.mean(s_log)
s_sum_avg = s_sum / np.mean(s_sum)
s_log_sum = (s_log_avg + s_sum_avg) / 2.0
accs_ = accs[~np.isnan(s_log_sum)]
scores_ = s_log_sum[~np.isnan(s_log_sum)]
numnan = np.isnan(s_log_sum).sum()

tau_log_sum, p = stats.kendalltau(accs_[:max(accs.shape[0] - numnan, 1)], scores_[:max(accs.shape[0] - numnan, 1)])
print("tau_log_sum:", tau_log_sum)

dict_log_sum_parts = {}
dict_log_sum_accs = {}
# s_log_sum_thre = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]   # 1.4-1.8
# s_log_thre = [0.8, 0.9, 1.0, 1.1]   # 0.8-1.2
# s_sum_thre = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]   # 0.1-2.7
# s_auc_thre = [0.56, 0.58, 0.60, 0.62, 0.64, 0.66]

# s_log_sum_thre = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]   # 0.4-1.8
# s_log_thre = [0.7, 0.8, 0.9, 1.0]   # 0.7-1.2
# s_sum_thre = [0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]   # 0.0-2.6
# s_auc_thre = [0.56, 0.58, 0.60, 0.62, 0.64, 0.66]  # 0.5 0.7

# 201-100
s_log_sum_thre = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]   # 0.4-1.8
s_log_thre = [0.7, 0.8, 0.9, 1.0]   # 0.7-1.2
s_sum_thre = [0, 0.2,0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]   # 0.-2,5
s_auc_thre = [0.87, 0.88, 0.89, 0.90]  # 0.87 0.91
for key in s_log_sum_thre:
    dict_log_sum_accs[key] = []
    dict_log_sum_parts[key] = []
for i in range(s_log_sum.shape[0]):
    for key in dict_log_sum_accs.keys():
        if s_log_sum[i] >= key:
            dict_log_sum_accs[key].append(accs[i])
            dict_log_sum_parts[key].append(s_log_sum[i])
for key in s_log_sum_thre:
    log_sum_np = np.array(dict_log_sum_parts[key])
    accs_np = np.array(dict_log_sum_accs[key])
    as_ = accs_np[~np.isnan(log_sum_np)]
    log_sum_ = log_sum_np[~np.isnan(log_sum_np)]
    numnan = np.isnan(log_sum_np).sum()
    tau, p = stats.kendalltau(as_[:max(accs_np.shape[0] - numnan, 1)], log_sum_[:max(accs_np.shape[0] - numnan, 1)])
    print("log_sum, num, thre, tau, acc:", log_sum_np.shape[0], key, tau, np.mean(accs_np))


dict_log_parts = {}
dict_log_accs = {}

for key in s_log_thre:
    dict_log_accs[key] = []
    dict_log_parts[key] = []
for i in range(s_log_avg.shape[0]):
    for key in dict_log_parts.keys():
        if s_log_avg[i] >= key:
            dict_log_accs[key].append(accs[i])
            dict_log_parts[key].append(s_log_avg[i])
for key in s_log_thre:
    log_np = np.array(dict_log_parts[key])
    accs_np = np.array(dict_log_accs[key])
    as_ = accs_np[~np.isnan(log_np)]
    log_ = log_np[~np.isnan(log_np)]
    numnan = np.isnan(log_np).sum()
    tau, p = stats.kendalltau(as_[:max(accs_np.shape[0] - numnan, 1)], log_[:max(accs_np.shape[0] - numnan, 1)])
    print("log, num, thre, tau, acc:",log_np.shape[0], key, tau, np.mean(accs_np))




dict_sum_parts = {}
dict_sum_accs = {}

for key in s_sum_thre:
    dict_sum_accs[key] = []
    dict_sum_parts[key] = []
for i in range(s_sum_avg.shape[0]):
    for key in dict_sum_accs.keys():
        if s_sum_avg[i] >= key:
            dict_sum_accs[key].append(accs[i])
            dict_sum_parts[key].append(s_sum_avg[i])
for key in s_sum_thre:
    sum_np = np.array(dict_sum_parts[key])
    accs_np = np.array(dict_sum_accs[key])
    as_ = accs_np[~np.isnan(sum_np)]
    sum_ = sum_np[~np.isnan(sum_np)]
    numnan = np.isnan(sum_np).sum()
    tau, p = stats.kendalltau(as_[:max(accs_np.shape[0] - numnan, 1)], sum_[:max(accs_np.shape[0] - numnan, 1)])
    print("sum, num, thre, tau, acc:",sum_np.shape[0], key, tau, np.mean(accs_np))







#    auc   acc      tau

dict_auc_parts = {}
dict_auc_score = {}
for key in s_auc_thre:
    dict_auc_parts[key] = []
    dict_auc_score[key] = []
for i in range(s_auc.shape[0]):
    for key in dict_auc_parts.keys():
        if s_auc[i] >= key:
            dict_auc_parts[key].append(s_auc[i])
            dict_auc_score[key].append(accs[i])
for key in s_auc_thre:
    accs_np = np.array(dict_auc_score[key])
    auc_np = np.array(dict_auc_parts[key])

    as_ = accs_np[~np.isnan(auc_np)]
    auc_ = auc_np[~np.isnan(auc_np)]
    numnan = np.isnan(auc_np).sum()
    tau_auc, p = stats.kendalltau(as_[:max(accs_np.shape[0] - numnan, 1)], auc_[:max(accs_np.shape[0] - numnan, 1)])
    print("auc-ass, num, thre, tau, acc:", auc_np.shape[0], key, tau_auc, np.mean(accs_np))

# 不要计算一个狭窄范围内的相关性，肯定很低，难以比较，没有意义
