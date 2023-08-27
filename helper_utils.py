import numpy as np


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def asymmetricKL(P, Q):
    return np.sum(P * np.log((P + 1e-12) / (Q + 1e-12) + 1e-12), axis=1)  # Calculate the kl-divergence between P and Q


def comput_similarity(a, b):  # (f,) (f,)
    a = a / np.sqrt(np.sum(np.square(a)))
    b = b / np.sqrt(np.sum(np.square(b)))
    return np.sum(a * b)


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))
