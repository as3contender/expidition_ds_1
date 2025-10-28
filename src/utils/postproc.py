# utils/postproc.py
import numpy as np
import cv2


def postproc_per_class(prob_stack, thresholds):
    """
    prob_stack: [C,H,W] float in [0,1]
    returns bin_stack: [C,H,W] uint8
    """
    C, H, W = prob_stack.shape
    out = np.zeros((C, H, W), dtype=np.uint8)
    for ci in range(C):
        thr = thresholds.get(ci, 0.35)
        binm = (prob_stack[ci] >= thr).astype(np.uint8)

        # Пример: усилить дороги/караванные пути морфологией
        # Допустим, индексы классов известны:
        # map_cls = {'dorogi': d_idx, 'karavannye_puti': k_idx}
        # тут для примера применим ко всем тонким:
        k = np.ones((3, 3), np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k, iterations=1)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k, iterations=1)

        out[ci] = binm
    return out
