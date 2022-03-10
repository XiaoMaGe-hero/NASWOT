import numpy as np
import random
amount = 8000
s_log = np.load('results/201_s.npy')
s_log = s_log[:amount]
s_sum = np.load('results/201_s_sum.npy')
s_sum = s_sum[:amount]
s_sum = s_sum[~np.isinf(s_log)]
s_auc = np.load('results/201_s_auv.npy')
s_auc = s_auc[:amount]
s_auc = s_auc[~np.isinf(s_log)]
accs = np.load('results/201_acc.npy')
accs = accs[:amount]
accs = accs[~np.isinf(s_log)]

sample_size = 100
n_runs = 10

res = []
for i in range(n_runs):
    ind = random.sample(range(0, 7000), sample_size)
    accs_ind = accs[ind].tolist()

    res.append(accs[accs_ind.index(max(accs_ind))])
res = np.array(res)
print("random, mena, std, sample 10, run 10", np.mean(res), np.var(res))

res = []
for i in range(n_runs):
    ind = random.sample(range(0, 7000), sample_size)
    s_log_ind = s_log[ind].tolist()
    res.append(accs[ind[s_log_ind.index(max(s_log_ind))]])
res = np.array(res)
print("nawot, mena, std, sample 10, run 10", np.mean(res), np.var(res))

res = []
for i in range(n_runs):
    ind = random.sample(range(0, 7000), sample_size)
    s_log_ind = s_sum[ind].tolist()
    res.append(accs[ind[s_log_ind.index(max(s_log_ind))]])
res = np.array(res)
print("nawot-v1, mena, std, sample 10, run 10", np.mean(res), np.var(res))


res = []
for i in range(n_runs):
    ind = random.sample(range(0, 7000), sample_size)
    s_log_ind = s_log[ind]
    s_sum_ind = s_sum[ind]
    s_log_ind = (s_log_ind - np.min(s_log_ind)) / (np.max(s_log_ind) - np.min(s_log_ind) + 2)
    s_sum_ind = (s_sum_ind - np.min(s_sum_ind)) / (np.max(s_sum_ind) - np.min(s_sum_ind) + 2)
    s_sum_log = (s_log_ind + s_sum_ind) / 2.0
    s_sum_log = s_sum_log.tolist()
    res.append(accs[ind[s_sum_log.index(max(s_sum_log))]])
res = np.array(res)
print("nawot-v2, mena, std, sample 10, run 10", np.mean(res), np.var(res))





