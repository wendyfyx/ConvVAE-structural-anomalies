import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def f_linear(x, a):
    return a*x

def f_linear_c(x, a, b):
    return a*x + b

def get_fitted_line(x, coeff, fit_intercept=True):
    '''Get y given a fitted line and x'''
    if fit_intercept:
        return f_linear_c(x, *coeff)
    return f_linear(x, *coeff)

def fit_line(x, y, fit_intercept=True):
    '''Get coefficient and R2 for line fit'''
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    if fit_intercept:
        f = f_linear_c
    else:
        f = f_linear
    popt, _ = curve_fit(f, x, y)
    y_pred = get_fitted_line(x, popt, fit_intercept=fit_intercept)
    return popt, r2_score(y, y_pred)