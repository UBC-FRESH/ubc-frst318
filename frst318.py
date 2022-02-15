import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def discount(v, n, i):
    return v * ((1 + i)**-n)

def compound(v, n, i):
    return v * ((1 + i)**+n)

def plot_v(v=1., n=10, i=0.05, mode='c', figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    if mode == 'c':
        _v = compound(v, n, i)
        ax.bar(0, v, label='reference value')
        ax.bar(n, _v, label='compounded value')
    else:
        _v = discount(v, n, i)
        ax.bar(n, v, label='reference value')
        ax.bar(0, _v, label='discounted value')
    plt.legend()
    ax.set_xlabel('n')
    ax.set_ylabel('V')
    plt.show()
    print('i', i)
    print('projected value', _v)
    
def npv(X, V, i):
    I = [i for x in X]
    return sum(np.vectorize(discount)(V, X, I))

def plot_cashflow(X, V, i=0.05, figsize=(12, 6), x_scale=1.):
    _npv = npv(X, V, i)
    _V = np.vectorize(discount)(V, X, [i for x in X])
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(X * x_scale, V, alpha=0.5, label='Cashflow')
    ax.bar(X * x_scale, _V, alpha=0.5, label='Cashflow')    
    ax.bar(0, _npv, alpha=0.5, label='NPV')
    plt.axhline(0.)
    ax.set_xlabel('Period')
    ax.set_ylabel('V')
    plt.xticks([i for i in range(int(X[-1]*x_scale+1))])
    plt.legend()
    plt.show()
    print('NPV at i=%0.3f: $%0.2f' % (i, _npv))
    
def pv_pps(a, i, t):
    return a * (1 / (pow(1 + i, t) - 1))

def pv_tps(a, i, t, n):
    return a * ((pow(1 + i, n*t) - 1) / ((pow(1 + i, t) - 1) * pow(1 + i, n*t)))

def lev_fvrot1(R, E, A, I, Y, P, Ch, r):
    result = -E * pow(1 + r, R)
    result += sum(I[t] * pow(1 + r, R - t) for t in I)
    result += A * (pow(1 + r, R) - 1) / r
    result += sum(P[p] * Y[p][R] - Ch for p in P)
    #result /= pow(1 + r, R) - 1
    return result

def plot_lev(R, E, A, I, Y, P, Ch, r, N):
    _lev_fvrot1 = lev_fvrot1(R, E, A, I, Y, P, Ch, r)
    X = np.array([(i + 1) * R for i in range(N)])
    V = np.array([_lev_fvrot1 for i in range(N)])
    plot_cashflow(X, V, i=r, x_scale=pow(R, -1))
    lev1 = pv_pps(_lev_fvrot1, r, R)
    lev2 = pv_tps(_lev_fvrot1, r, R, N)
    print('LEV: $%0.2f' % lev1)
    print('relative truncation error: %0.2f%%' % (100. * (lev2 - lev1) / lev1))
  