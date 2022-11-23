import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import optimize

from . import utils
from . import show_prints
from .show_prints import print

def CI_from_sigma(sigma):
    np.array(sigma)
    return 2*stats.norm.cdf(sigma)-1

def sigma_from_CI(CI):
    np.array(CI)
    return stats.norm.ppf(CI/2+1/2)

def delta_chi2_from_CI(CI, dof=1):
    np.array(CI)
    return stats.chi2.ppf(CI, dof)

def ks_w(data1, data2, wei1=None, wei2=None):
    """ Weighted Kolmogorov-Smirnov test. Returns the KS statistics and the p-value (in the limit of large samples).
    """
    if wei1 is None:
        wei1 = np.ones(len(data1))
    if wei2 is None:
        wei2 = np.ones(len(data2))
    wei1 = np.array(wei1)
    wei2 = np.array(wei2)
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    n1 = len(data1)
    n2 = len(data2)
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right').tolist()]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right').tolist()]
    d = np.max(np.abs(cdf1we - cdf2we))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    prob = stats.distributions.kstwobign.sf(en * d)
    return [d, prob]

def sort_consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def HPDI(data, intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True):
    intervals = np.sort(np.array([intervals]).flatten())
    if weights is None:
        weights = np.ones(len(data))
    weights = np.array(weights)
    counter = 0
    results = {}
    for interval in intervals:
        counts, bins = np.histogram(data, nbins, weights=weights, density=True)
        #counts, bins = hist
        nbins_val = len(counts)
        if print_hist:
            integral = counts.sum()
            plt.step(bins[:-1], counts/integral, where='post',color='green', label=r"train")
            plt.show()
        binwidth = bins[1]-bins[0]
        arr0 = np.transpose(np.concatenate(([counts*binwidth], [(bins+binwidth/2)[0:-1]])))
        arr0 = np.transpose(np.append(np.arange(nbins_val),np.transpose(arr0)).reshape((3, nbins_val)))
        arr = np.flip(arr0[arr0[:, 1].argsort()], axis=0)
        q = 0
        bin_labels = np.array([])
        for i in range(nbins_val):
            if q <= interval:
                q = q + arr[i, 1]
                bin_labels = np.append(bin_labels, arr[i, 0])
            else:
                bin_labels = np.sort(bin_labels)
                result = [[arr0[tuple([int(k[0]), 2])], arr0[tuple([int(k[-1]), 2])]] for k in sort_consecutive(bin_labels)]
                result_previous = result
                binwidth_previous = binwidth
                if optimize_binning:
                    while (len(result) == 1 and nbins_val+nbins < np.sqrt(len(data))):
                        nbins_val = nbins_val+nbins
                        result_previous = result
                        binwidth_previous = binwidth
                        #nbins_val_previous = nbins_val
                        HPD_int_val = HPDI(data, intervals=interval, weights=weights, nbins=nbins_val, print_hist=False)
                        result = HPD_int_val[interval]["Intervals"]
                        binwidth = HPD_int_val[interval]["Bin width"]
                break
        #results.append([interval, result_previous, nbins_val, binwidth_previous])
        results[interval] = {"Probability": interval, "Intervals": result_previous, "Number of bins": nbins_val, "Bin width": binwidth_previous}
        counter = counter + 1
    return results

def HPDI_error(HPDI):
    res = {}
    different_lengths = False
    for key_par, value_par in HPDI.items():
        dic = {}
        for sample in value_par['true'].keys():
            true = value_par['true'][sample]
            pred = value_par['pred'][sample]
            dic2 = {}
            for CI in true.keys():
                dic3 = {"Probability": true[CI]["Probability"]}
                if len(true[CI]["Intervals"])==len(pred[CI]["Intervals"]):
                    dic3["Absolute error"] = (np.array(true[CI]["Intervals"])-np.array(pred[CI]["Intervals"])).tolist()
                    dic3["Relative error"] = ((np.array(true[CI]["Intervals"])-np.array(pred[CI]["Intervals"]))/(np.array(true[CI]["Intervals"]))).tolist()
                else:
                    dic3["Absolute error"] = None
                    dic3["Relative error"] = None
                    different_lengths = True
                dic2 = {**dic2, **{CI: dic3}}
            dic = {**dic, **{sample: dic2}}
        res = {**res, **{key_par: dic}}
        if different_lengths:
            print("For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.")
    return res

def HPD_quotas(data, intervals=0.68, weights=None, nbins=25, from_top=True):
    intervals = np.sort(np.array([intervals]).flatten())
    counts, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=nbins, range=None, normed=None, weights=weights, density=None)
    #counts, binsX, binsY = np.histogram2d(data[:, 0], data[:, 1], bins=nbins, range=None, normed=None, weights=weights, density=None)
    integral = counts.sum()
    counts_sorted = np.flip(np.sort(utils.flatten_list(counts)))
    quotas = intervals
    q = 0
    j = 0
    for i in range(len(counts_sorted)):
        if q < intervals[j] and i < len(counts_sorted)-1:
            q = q + counts_sorted[i]/integral
        elif q >= intervals[j] and i < len(counts_sorted)-1:
            if from_top:
                quotas[j] = 1-counts_sorted[i]/counts_sorted[0]
            else:
                quotas[j] = counts_sorted[i]/counts_sorted[0]
            j = j + 1
        else:
            for k in range(j, len(intervals)):
                quotas[k] = 0
            j = len(intervals)
        if j == len(intervals):
            return quotas

def weighted_quantiles(data, quantiles=0.68, weights=None, data_sorted=False, onesided=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    
        - param data numpy.array with data
        - param quantiles array-like with many quantiles needed
        - param weights array-like of the same length as `array`
        - param data_sorted bool, if True, then will avoid sorting of initial array
        - return numpy.array with computed quantiles.
    """
    quantiles = np.sort(np.array([quantiles]).flatten())
    if onesided:
        data = np.array(data[data > 0])
    else:
        data = np.array(data)
    if weights is None:
        weights = np.ones(len(data))
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'
    if not data_sorted:
        sorter = np.argsort(data)
        data = data[sorter]
        weights = weights[sorter]
    w_quantiles = np.cumsum(weights) - 0.5 * weights
    w_quantiles -= w_quantiles[0]
    w_quantiles /= w_quantiles[-1]
    result = np.transpose(np.concatenate((quantiles, np.interp(quantiles, w_quantiles, data))).reshape(2, len(quantiles))).tolist()
    return result

def weighted_central_quantiles(data, intervals=0.68, weights=None, onesided=False):
    intervals = np.sort(np.array([intervals]).flatten())
    if onesided:
        data = np.array(data[data > 0])
    else:
        data = np.array(data)
    return [[i, [weighted_quantiles(data, (1-i)/2, weights), weighted_quantiles(data, 0.5, weights), weighted_quantiles(data, 1-(1-i)/2, weights)]] for i in intervals]

def compute_maximum_logpdf(logpdf=None, 
                           ndims=None, 
                           pars_init=None, 
                           pars_bounds=None,
                           optimizer = {},
                           minimization_options={},
                           verbose=True):
    """
    """
    if verbose < 0:
        verbose_sub = 0
    else:
        verbose_sub = verbose
    if logpdf is None:
        raise Exception("The 'logpdf' input argument cannot be empty.")
    def minus_logpdf(x): return -logpdf(x)
    if ndims == None and pars_init is not None:
        ndims = len(pars_init)
    elif ndims != None and pars_init is None:
        pars_init = np.full(ndims, 0)
    elif ndims == None and pars_init is None:
        print("Please specify npars or pars_init or both", show = verbose)
    utils.check_set_dict_keys(optimizer, ["name",
                                          "args",
                                          "kwargs"],
                                         ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
    args = optimizer["args"]
    kwargs = optimizer["kwargs"]
    options = minimization_options
    if pars_bounds is None:
        #print("Optimizing")
        ml = optimize.minimize(minus_logpdf, pars_init, *args, options=options, **kwargs)
    else:
        #print("Optimizing")
        pars_bounds = np.array(pars_bounds)
        bounds = optimize.Bounds(pars_bounds[:, 0], pars_bounds[:, 1])
        ml = optimize.minimize(minus_logpdf, pars_init, *args, bounds=bounds, options=options, **kwargs)
    return [ml['x'], -ml['fun']]

def compute_profiled_maximum_logpdf(logpdf=None, 
                                   pars=None,
                                   pars_val=None,
                                   ndims=None, 
                                   pars_init=None, 
                                   pars_bounds=None, 
                                   optimizer={},
                                   minimization_options={},
                                   verbose=True):
    """
    """
    if verbose < 0:
        verbose_sub = 0
    else:
        verbose_sub = verbose
    # Add check that pars are within bounds
    if logpdf is None:
        raise Exception("The 'logpdf' input argument cannot be empty.")
    if pars is None:
        raise Exception("The 'pars' input argument cannot be empty.")
    if pars_val is None:
        raise Exception("The 'pars_val' input argument cannot be empty.")
    if len(pars)!=len(pars_val):
        raise Exception("The input arguments 'pars' and 'pars_val' should have the same length.")
    pars = np.array(pars)
    pars_insert = pars - range(len(pars))
    if ndims == None and pars_init is not None:
        ndims = len(pars_init)
    elif ndims != None and pars_init is None:
        pars_init = np.full(ndims, 0)
    elif ndims == None and pars_init is None:
        print("Please specify ndims or pars_init or both",show=verbose())
    else:
        if len(pars_init)!=ndims:
            raise Exception("Parameters initialization has the wrong dimension. The dimensionality should be"+str(ndims)+".")
    pars_init_reduced = np.delete(pars_init, pars)
    utils.check_set_dict_keys(optimizer, ["name",
                                          "args",
                                          "kwargs"],
                                         ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
    args = optimizer["args"]
    kwargs = optimizer["kwargs"]
    options = minimization_options
    def minus_logpdf(x):
        return -logpdf(np.insert(x, pars_insert, pars_val))
    if pars_bounds is not None:
        pars_bounds=np.array(pars_bounds)
        if len(pars_bounds)!=len(pars_init):
            raise Exception("The input argument 'pars_bounds' should be either 'None' or have the same length of 'pars'.")
        if not ((np.all(pars_val >= pars_bounds[pars, 0]) and np.all(pars_val <= pars_bounds[pars, 1]))):
            print("Parameter values",pars_val,"lies outside parameters bounds",pars_bounds,".")
            return
    if pars_bounds is None:
        #print("Optimizing")
        ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, options=options, **kwargs)
    else:
        #print("Optimizing")
        pars_bounds_reduced = np.delete(pars_bounds, pars,axis=0)
        pars_bounds_reduced = np.array(pars_bounds_reduced)
        bounds=optimize.Bounds(pars_bounds_reduced[:, 0], pars_bounds_reduced[:, 1])
        ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, bounds=bounds, options=options, **kwargs)
    return [np.insert(ml['x'], pars_insert, pars_val, axis=0), -ml['fun']]

def compute_maximum_sample(X=None,
                           Y=None):
    """

    """
    X = np.array(X)
    Y = np.array(Y)
    y_max = np.amax(Y)
    pos_max = np.where(Y == y_max)[0][0]
    Y[pos_max] = -np.inf
    y_next_max = np.amax(Y)
    pos_next_max = np.where(Y == y_next_max)[0][0]
    x_max = X[pos_max]
    x_next_max = X[pos_next_max]
    return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max)]

def compute_profiled_maximum_sample(pars,
                                    pars_val,
                                    X=None,
                                    Y=None,
                                    binwidths="auto"):
    """

    """
    X = np.array(X)
    Y = np.array(Y)
    if type(binwidths) == float or type(binwidths) == int:
        binwidths = np.full(len(pars), binwidths)
    if binwidths != "auto":
        slicings = []
        for i in range(len(pars)):
            slicings.append([p > pars_val[i]-binwidths[i]/2 and p <
                             pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
        slicing = np.prod(np.array(slicings), axis=0).astype(bool)
        npoints = np.count_nonzero(slicing)
        y_max = np.amax(Y[slicing])
        pos_max = np.where(Y == y_max)[0][0]
        Y[pos_max] = -np.inf
        y_next_max = np.amax(Y[slicing])
        pos_next_max = np.where(Y == y_next_max)[0][0]
        x_max = X[pos_max]
        x_next_max = X[pos_next_max]
    elif binwidths == "auto":
        binwidths = np.full(len(pars), 0.001)
        npoints = 0
        while npoints < 2:
            binwidths = binwidths+0.001
            slicings = []
            for i in range(len(pars)):
                slicings.append([p > pars_val[i]-binwidths[i]/2 and p <
                                 pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
            slicing = np.prod(np.array(slicings), axis=0).astype(bool)
            npoints = np.count_nonzero(slicing)
        y_max = np.amax(Y[slicing])
        pos_max = np.where(Y == y_max)[0][0]
        Y[pos_max] = -np.inf
        y_next_max = np.amax(Y[slicing])
        pos_next_max = np.where(Y == y_next_max)[0][0]
        x_max = X[pos_max]
        x_next_max = X[pos_next_max]
    return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max), npoints]
