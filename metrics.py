from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from six import iteritems


def bootstrap(metric, r_pred, r_true, user_ids, item_ids):
    n = len(r_true)
    values = []
    for i in range(1000):
        x = np.random.choice(n, size=n)
        values.append(metric(r_pred[x], r_true[x], user_ids[x], item_ids[x]))
    return values

def rmse(r_pred, r_true, user_ids, item_ids):
    return np.mean((r_pred - r_true)**2)**0.5

def mae(r_pred, r_true, user_ids, item_ids):
    return np.mean(np.abs(r_pred - r_true))

def spearman(r_pred, r_true, user_ids, item_ids):
    return spearmanr(r_pred, r_true)[0]

def fcp(r_pred, r_true, user_ids, item_ids):
    # https://github.com/NicolasHug/Surprise/blob/d29b255826506c95c4822fe633f1107354c3f6a5/surprise/accuracy.py
    predictions = defaultdict(list)
    for i in range(len(user_ids)):
        predictions[int(user_ids[i])].append([r_true[i], r_pred[i]])
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)
    for u0, preds in iteritems(predictions):
        if len(preds) == 1:
            continue
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1
    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0
    return nc / (nc + nd)

def bpr(r_pred, r_true, user_ids, item_ids):
    # rescale predictions to range 1-5
    r_min, r_range = r_true.min(), r_true.max() - r_true.min()
    r_pred, r_true = (r_pred - r_min)/r_range*4 + 1, (r_true - r_min)/r_range*4 + 1
    # group input/output pairs by user_id
    groups = {}
    for i, user_id in enumerate(user_ids):
        if user_id in groups:
            groups[user_id].append((r_pred[i], r_true[i]))
        else:
            groups[user_id] = [(r_pred[i], r_true[i])]
    # compute bpr
    total, count = 0, 0
    for user_id, group in groups.items():
        for i in range(1, len(group)):
            for j in range(i):
                r_pred_i, r_true_i = group[i]
                r_pred_j, r_true_j = group[j]
                x = r_pred_i - r_pred_j if r_true_i > r_true_j else r_pred_j - r_pred_i
                total, count = total + np.log(1/(1 + np.exp(-x))), count + 1
    # normalize bpr by count and express as probability
    return np.exp(total/count)
