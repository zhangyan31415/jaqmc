# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Independent-walker statistics adapted from JAQMC commit 266542b.

Energy and charge stationarity must both pass; low Rhat alone cannot exclude
metastability in a motif-seeded variational state.
"""
import numpy as np
from scipy.stats import norm, rankdata


def batch_mean_se(x, block):
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    count = len(x)//block
    if count < 2:
        return float("nan")
    batches = x[:count*block].reshape(count, block, x.shape[1]).mean(axis=1)
    variance = np.mean(np.var(batches, axis=0, ddof=1))
    return float(np.sqrt(variance/(count*x.shape[1])))


def split_rhat(x):
    x = np.asarray(x, dtype=float).reshape(len(x), -1).T
    half = x.shape[1]//2
    split = np.concatenate([x[:, :half], x[:, -half:]])
    def calc(v):
        within = np.mean(np.var(v, axis=1, ddof=1))
        between = v.shape[1]*np.var(v.mean(axis=1), ddof=1)
        if within < 1e-28:
            return 1. if between < 1e-28 else float("inf")
        return float(np.sqrt(((v.shape[1]-1)*within+between)/v.shape[1]/within))
    def ranked(v):
        r = rankdata(v.ravel()).reshape(v.shape)
        return norm.ppf((r-.375)/(v.size+.25))
    return dict(raw=calc(split), rank=max(calc(ranked(split)),
        calc(ranked(abs(split-np.median(split))))))


def summary(x):
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    count, walkers = x.shape
    centered = x-x.mean(axis=0)
    frequency = np.fft.rfft(centered, n=2*count, axis=0)
    covariance = np.fft.irfft(frequency*frequency.conj(),
        n=2*count, axis=0)[:count].mean(axis=1)/np.arange(count, 0, -1)
    variance = float(covariance[0])
    if variance < 1e-28:
        tau, window = .5, 0
    else:
        rho = covariance/variance
        pairs = rho[:count//2*2].reshape(-1, 2).sum(axis=1)
        stopped = np.flatnonzero(pairs <= 0)
        if len(stopped):
            pairs = pairs[:stopped[0]]
        tau = max(1e-8, float(-.5+np.minimum.accumulate(pairs).sum()))
        window = 2*len(pairs)
    blocks = [2**k for k in range(int(np.log2(count))-1)]
    return dict(mean=float(x.mean()),
        independent_walker_se=float(x.mean(axis=0).std(ddof=1)/np.sqrt(walkers)),
        ac=dict(tau=tau, variance=variance, window=window,
                se=float(np.sqrt(2*tau*variance/(count*walkers)))),
        rhat=split_rhat(x),
        block_se={str(b): batch_mean_se(x, b) for b in blocks})
