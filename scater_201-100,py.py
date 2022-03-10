import numpy as np
import random
import matplotlib.pyplot as plt
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
s_log = s_log[~np.isinf(s_log)]

s_log_avg = (s_log - s_log.min()) / (s_log.max() - s_log.min())
s_sum_avg = (s_sum - s_sum.min()) / (s_sum.max() - s_sum.min())
s_log_sum = (s_log_avg + s_sum_avg) / 2.0

fig = plt.figure(figsize=[5, 3.3])
ind = random.sample(range(0, 7000), 500)
ax1 = fig.add_subplot(221)
ax1.scatter(accs[ind], s_log[ind], color='green', marker='.')
ax1.set(xlabel='Aalidation Accuracy', ylabel='Score', title=r'NASWOT, $\tau = 0.60$  ')
ax2 = fig.add_subplot(222)
ax2.scatter(accs[ind], s_sum[ind], color='red', marker='.')
ax2.set(xlabel='Aalidation Accuracy', ylabel='Score', title=r'NASWOT-V1, $\tau = 0.61$')
ax3 = fig.add_subplot(223)
ax3.scatter(accs[ind], s_log_sum[ind], color='blue', marker='.')
ax3.set(xlabel='Aalidation Accuracy', ylabel='Score', title=r'NASWOT-V2, $\tau = 0.53$')
ax4 = fig.add_subplot(224)
ax4.scatter(accs[ind], s_auc[ind], color='yellow', marker='.')
ax4.set(xlabel='Aalidation Accuracy', ylabel='Score', title=r'SAUC, $\tau = 0.30$ ')
fig.tight_layout()
plt.show()

