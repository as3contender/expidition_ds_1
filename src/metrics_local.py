import numpy as np
import torch


def f2_weighted(y_true, y_prob, thresholds, class_names, class_weights_dict):
    y = y_true.detach().cpu().numpy().astype(np.uint8)
    p = y_prob.detach().cpu().numpy().astype(np.float32)
    C = y.shape[1]
    f2s = []
    for k, cname in enumerate(class_names):
        thr = thresholds.get(cname, 0.35)
        pred = (p[:, k] >= thr).astype(np.uint8)
        tp = (pred & y[:, k]).sum()
        fp = (pred & (1 - y[:, k])).sum()
        fn = ((1 - pred) & y[:, k]).sum()
        P = tp / (tp + fp + 1e-9)
        R = tp / (tp + fn + 1e-9)
        f2 = (5 * P * R) / (4 * P + R + 1e-9)
        f2s.append(f2)
    f2s = np.array(f2s, dtype=np.float32)
    w = np.array([class_weights_dict[c] for c in class_names], dtype=np.float32)
    w = w / (w.sum() + 1e-9)
    return float((w * f2s).sum()), f2s
