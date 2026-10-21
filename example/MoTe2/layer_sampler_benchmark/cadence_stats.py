"""Whole-walker bootstrap and retrospective precision crossings (NumPy only)."""
import numpy as np


def bootstrap_indices(seeds,walkers,repeats,random_seed):
    rng=np.random.default_rng(random_seed)
    return (rng.integers(walkers,size=(repeats,seeds,walkers))+
            walkers*np.arange(seeds)[None,:,None]).reshape(repeats,seeds*walkers)


def replicate_bootstrap(means,repeats=4000,random_seed=71022):
    means=np.asarray(means)
    idx=bootstrap_indices(*means.shape,repeats,random_seed)
    return means.ravel()[idx].var(axis=1,ddof=1)/means.size


def select_cadences(summaries):
    return {method:int(min(cases,key=lambda k:(cases[k]['cost'],int(k))))
            for method,cases in summaries.items()}


def stable_crossings(se_draws,times,target):
    """Censor beyond observed range; never extrapolate a crossing."""
    se=np.asarray(se_draws);times=np.asarray(times)
    good=np.maximum.accumulate(se[:,::-1],axis=1)[:,::-1]<=target
    found=good.any(axis=1);first=good.argmax(axis=1)
    values=np.where(found,times[first],np.inf)
    # Inverse empirical CDF preserves right-censoring, including upper quantile.
    ordered=np.sort(values)
    bounds=[float(ordered[min(len(ordered)-1,int(np.ceil(q*len(ordered)))-1)]) for q in [.025,.975]]
    return dict(crossing_seconds=[float(v) if np.isfinite(v) else None for v in values],
                censored_fraction=float((~found).mean()),
                left_censored_at_first_grid_fraction=float(np.mean(found & (first==0))),
                interval_95_seconds=[v if np.isfinite(v) else None for v in bounds])
