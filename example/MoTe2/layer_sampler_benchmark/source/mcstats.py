"""Uncertainty estimators with independent walkers kept as independent replicates."""
import numpy as np
from scipy.stats import norm, rankdata, f as f_distribution


def as_chains(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(x.shape[0], -1)


def independent_mean_se(x):
    x = as_chains(x)
    means = x.mean(axis=0)
    return float(means.std(ddof=1)/np.sqrt(len(means)))


def batch_mean_se(x, block):
    x = as_chains(x)
    nblock = len(x)//block
    if nblock < 2:
        return float('nan')
    batches = x[:nblock*block].reshape(nblock, block, x.shape[1]).mean(axis=1)
    # Pool independent walker variance estimates, rather than having only
    # eight aggregate blocks at the longest block length.
    variance = np.mean(np.var(batches, axis=0, ddof=1))
    return float(np.sqrt(variance/(nblock*x.shape[1])))


def bartlett_curve(x, bandwidths):
    """Positive lag-window spectral estimator; no reversibility requirement.

    Still requires stationary mixing chains. Finite bandwidth can miss slow
    tails, so agreement with block/replicate estimators must be checked.
    """
    x = as_chains(x)
    T, M = x.shape
    centered = x-x.mean(axis=0)
    frequency = np.fft.rfft(centered, n=2*T, axis=0)
    # Biased sample autocovariances yield a positive-semidefinite Bartlett
    # spectral estimate (up to floating-point cancellation).
    ac = np.fft.irfft(frequency*frequency.conj(), n=2*T, axis=0)[:T].mean(axis=1)/T
    result = {}
    for b in bandwidths:
        b = min(int(b), T-1)
        lags = np.arange(1, b+1)
        gamma = max(0., float(ac[0]+2*np.sum((1-lags/(b+1))*ac[1:b+1])))
        result[str(b)] = dict(gamma=gamma, se=float(np.sqrt(gamma/(T*M))))
    return result


def bartlett_se(x, bandwidth):
    return bartlett_curve(x, [bandwidth])[str(min(bandwidth, len(x)-1))]['se']


def integrated_time(x):
    x = as_chains(x)
    T, M = x.shape
    centered = x-x.mean(axis=0)
    freq = np.fft.rfft(centered, n=2*T, axis=0)
    ac = np.fft.irfft(freq*freq.conj(), n=2*T, axis=0)[:T].mean(axis=1)
    ac /= np.arange(T, 0, -1)
    variance = float(ac[0])
    if variance < 1e-28:
        return dict(tau=.5, variance=variance, gamma=variance, se=0., window=0,
                    rho=[1.]+[0.]*min(T-1, 127))
    rho = ac/variance
    pairs = rho[:T//2*2].reshape(-1, 2).sum(axis=1)
    stopped = np.where(pairs <= 0)[0]
    if len(stopped):
        pairs = pairs[:stopped[0]]
    monotone = np.minimum.accumulate(pairs)
    # Do not impose tau>=1/2: genuine negative correlations may help Wolff.
    tau = max(1e-8, float(-.5+monotone.sum()))
    gamma = 2*tau*variance
    return dict(tau=tau, variance=variance, gamma=gamma,
                se=float(np.sqrt(gamma/(T*M))), window=int(2*len(pairs)),
                rho=rho[:min(T, 128)].tolist())


def split_rhat(x):
    x = as_chains(x).T
    T = x.shape[1]
    half = T//2
    split = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    def calc(v):
        within = np.mean(np.var(v, axis=1, ddof=1))
        between = v.shape[1]*np.var(np.mean(v, axis=1), ddof=1)
        if within < 1e-28:
            return 1. if between < 1e-28 else float('inf')
        return float(np.sqrt(((v.shape[1]-1)/v.shape[1]*within+between/v.shape[1])/within))
    def ranked(v):
        r = rankdata(v.ravel()).reshape(v.shape)
        return norm.ppf((r-.375)/(v.size+.25))
    return dict(raw=calc(split), rank=max(calc(ranked(split)),
        calc(ranked(abs(split-np.median(split))))))


def seed_anova(x):
    """x has shape time,seed,walker; tests excess seed-level mean dispersion."""
    means = np.asarray(x).mean(axis=0)
    seeds, walkers = means.shape
    within = np.var(means, axis=1, ddof=1).mean()
    between = walkers*np.var(means.mean(axis=1), ddof=1)
    ratio = float(between/max(within, 1e-30))
    return dict(F=ratio, p=float(f_distribution.sf(ratio, seeds-1, seeds*(walkers-1))),
                seed_se=float(means.mean(axis=1).std(ddof=1)/np.sqrt(seeds)))


def stable_precision_crossing(grid, delta, burn_seconds=0.):
    """Retrospective observed threshold; not an optional-stopping guarantee."""
    prefixes=sorted(map(int,grid))
    for j,n in enumerate(prefixes):
        if all(grid[str(k)]['se_range'][1]<=delta for k in prefixes[j:]):
            return dict(prefix=n,energy_seconds=grid[str(n)]['energy_seconds'],
                with_burn_seconds=grid[str(n)]['energy_seconds']+burn_seconds,
                previous_prefix=prefixes[j-1] if j else None,
                previous_energy_seconds=grid[str(prefixes[j-1])]['energy_seconds'] if j else 0.)
    return None


def summary(x):
    x = as_chains(x)
    ac = integrated_time(x)
    blocks = [2**k for k in range(0, int(np.log2(len(x)))-1)]
    return dict(mean=float(x.mean()), independent_walker_se=independent_mean_se(x),
        ac=ac, rhat=split_rhat(x),
        bartlett=bartlett_curve(x, blocks),
        block_se={str(b):batch_mean_se(x, b) for b in blocks})
