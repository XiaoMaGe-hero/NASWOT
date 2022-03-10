import numpy as np
import torch


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)

    return ld


def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }


def get_score_func(score_name):
    return _scores[score_name]


def get_score_sum_func(K):
    return np.sum(K)


def get_auc_score_func(K, target_tesor):
    # K ndarray
    target_verbose = target_tesor.numpy()
    target_unique = np.unique(target_verbose)
    dict_label_val = {}
    for val in target_unique:
        dict_label_val[val] = []
    for i in range(K.shape[0]):
        label = target_verbose[i]
        for ind, val in enumerate(K[i]):
            if target_verbose[ind] == label:
                dict_label_val[label].append(val)
            else:
                dict_label_val[label].append(-val)
    auc_val = 0
    for val in target_unique:
        pos_inds = 0
        p_f = 0
        for i in range(len(dict_label_val[val])):
            if dict_label_val[val][i] > 0:
                pos_inds += 1
                for j in range(len(dict_label_val[val])):
                    if dict_label_val[val][j] <0 and -dict_label_val[val][j] < dict_label_val[val][i]:
                        p_f += 1
                    else:
                        pass
        auc_val += p_f / (pos_inds*( len(dict_label_val[val]) - pos_inds ))

    return auc_val / target_unique.shape[0]
