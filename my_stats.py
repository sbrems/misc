import numpy as np
from scipy.stats import f, gamma as scipygamma
from scipy.optimize import newton
from scipy.special import digamma
from statsmodels.nonparametric import kde


def weighted_avg_and_var(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    # Fast and numerically precise
    return (average, variance)


def weighted_variance(values, weights, ddof=0):
    """
    Return the weighted variance.
    values, weights: 1d np arrays with the same length
    ddof: number of delta degrees of freedom. Set to one for
    an unbiased estimator. default 0
    """
    assert values.shape == weights.shape
    average = np.average(values, weights=weights)
    if ddof == 0:
        return np.average((values - average)**2, weights=weights)
    else:
        return np.sum(weights*(values-average)**2) / (
            np.sum(weights)*(1.-1./len(values)))


def chisquared(func, xvals, yvals, errors=None):
    '''Calculate the sum of squares.
    If no errors are given, 1 is used'''
    assert len(xvals) == len(yvals)
    if errors is None:
        errors = np.full((len(xvals)), 1)
    chi2 = 0
    for xval, yval, err in zip(xvals, yvals, errors):
        chi2 += ((func(xval) - yval) / err)**2
    return chi2


def f_test(chi1, chi2, ndata, par1, par2, alpha=0.02, verbose=True):
    '''Do the f-test. Function from Sepideh. Returns 0 if less params is better,
    1 if null-hypothesis rejected and more params give gain'''
    # chi1_red = chi1 / (ndata - par1)
    chi2_red = chi2 / (ndata - par2)

    F = ((chi1 - chi2) / (par2 - par1)) / chi2_red  # Fstatistic
    # p-value threshold = 0.005
    alpha = 0.005  # 0.01 #Or whatever you want your alpha to be.

    p_value = f.sf(F, par2 - par1, ndata - par2, loc=0, scale=1)
    print("p_value", p_value)
    print("log(p_value)", np.log10(p_value))
    if p_value < alpha:
        better_model = 1
        print("Null hypothesis rejected. Model 1 with mre params is better.")
        print("Probability = ", (1.0 - p_value) * 100.0, '%')
    else:
        print("Null hypothesis cannot be rejected. \
        Seems model 0 with less params is good enough.")
        better_model = 0
    return better_model


def kde_smallest_interval(array, kernel="gau", area=68.27):
    '''Calculate the lower and upper limits of the array in such a way,
    area (default=68.27) of the values are in a as small as possible spread.
    For small arrays we need the kde.
    Returns the min and max bounds'''
    array = np.array(array)
    assert len(array.shape) == 1, "The array needs to be 1D to \
estimate the errors!"
    dens = kde.KDEUnivariate(array)
    dens.fit()
    icdf = dens.icdf

    npoints = icdf.shape[0]
    # make sure kde samples dense enough
    assert npoints >= 100

    # find the minimum interval now
    spread = np.int(np.round(npoints*(area/100.)))
    optimumidx = np.argmin(icdf[spread:] - icdf[:-spread])
    # return min/max bounds
    return icdf[optimumidx], icdf[optimumidx+spread]


def kde_confidence_intervals(array, kernel="gau"):
    '''Calculate the lower and upper limits of the array
    smoothing it first with a gaussian (default) kernel.
    input:
    array (1d np.array)
    The values to estimate the errors of
    kernel="gau"
    The kernel to use. Passes it to statsmodels, allowed values are:
    ”biw” for biweight
    ”cos” for cosine
    ”epa” for Epanechnikov
    ”gau” for Gaussian.
    ”tri” for triangular
    ”triw” for triweight
    ”uni” for uniform

    returns
    lowerlim, upperlim of the 68 percile
    '''
    array = np.array(array)
    assert len(array.shape) == 1
    dens = kde.KDEUnivariate(array)
    dens.fit()
    low = np.percentile(dens.icdf, 15.865)
    up = np.percentile(dens.icdf, 84.135)
    return low, up


def confid_intervals_given_mean(mean, array, pecentiles=[15.865, 84.135]):
    '''Give a mean/best guess and the array of values. Returns
    the percentiles values. If there is not enough datapoints left, nan is
    returned'''
    


class Gamma():
    '''Fit the gamma distribution. The main difference to scipy is,
    that it can also handle weights, but does not support an offset.
    Funktions:

    fit:
    fit the k and Theta values to the data. Optional: Give weights

    rvs:
    give k and theta (or determine them via fit before) and optionally size,
    to return size (default 1) random, gamma
    distributed values'''
    def __init__(self, k=None, theta=None):
        self.k = k
        self.theta = theta

    def fit(self, x, weights=None, return_vals=True):
        '''Fit gamma and theta'''
        self.x = x,
        self.weights = weights
        x = np.array(x)
        N = x.shape[0]
        if len(x.shape) != 1:
            raise ValueError('x has to be a 1dim np array.')
        if weights is None:
            sumxN = np.sum(x)/N
            sumlnxN = np.sum(np.log(x))/N
        else:
            weights = np.array(weights)
            if weights.shape != x.shape:
                raise ValueError('If weights are given they need the same \
shape as x')
            # weighted mean
            sumxN = np.sum(x*weights)*(x.shape[0]/np.sum(weights))/N
            sumlnxN = (np.sum(np.log(x*weights)) +
                       N*(np.log(N)-np.log((np.sum(weights)))))/N

        # solve for k first
        k0 = 1./(2*(np.log(sumxN) - sumlnxN))
        k = newton(lambda kk: np.log(sumxN) - sumlnxN -
                   np.log(kk) + digamma(kk),
                   x0=k0, maxiter=1000)
        theta = sumxN/k
        self.k = k
        self.theta = theta
        if return_vals:
            return k, theta

    def rvs(self, k=None, theta=None, size=1, loc=0):
        '''Return size gamma distributed values. if k and theta
        are not provided, assuming you did fit them before. With
        loc you can shift the distribution. This is not fitted!'''
        if k is None:
            k = self.k
            assert k is not None
        if theta is None:
            theta = self.theta
            assert theta is not None
        assert np.isfinite(k)
        assert np.isfinite(theta)
        # use scipy gamma to return the values
        return scipygamma.rvs(a=k, loc=loc, scale=theta, size=size)

