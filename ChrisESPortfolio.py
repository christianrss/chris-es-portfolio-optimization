#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from datetime import datetime

import multiprocessing
from multiprocessing import Pool
import yfinance as yf

start = '2010-01-01'
end = '2024-12-31'

tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']

benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05, 0])

df = yf.download(tickers, start=start, end=end, interval='1mo')
df = df['Close']
df = df.dropna().copy()


# In[2]:


print(df.head())


# In[3]:


return_tickers = []
for ticker in tickers:
    rtick = f"{ticker}_return"
    df[rtick] = df[ticker].pct_change()
    return_tickers.append(rtick)


# In[4]:


# split into train and test
n_train = int(0.8 * len(df))
df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]


# In[5]:


# pre-compute returns
train_returns = df_train[return_tickers].dropna().to_numpy()
test_returns = df_test[return_tickers].dropna().to_numpy()


# In[6]:


# add a column of 0s for cash
z = np.zeros((train_returns.shape[0], 1))
train_returns = np.hstack((train_returns, z))

z = np.zeros((test_returns.shape[0], 1))
test_returns = np.hstack((test_returns, z))


# In[ ]:




# In[7]:


def evolution_strategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters,
    pool):

    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)

        ## fast way
        R = pool.map(f, [params + sigma * N[j] for j in range(population_size)])
        R = np.array(R)

        m = R.mean()
        s = R.std()
        if s == 0:
            # we can't apply the following equation
            print("Skipping")
            continue

        A = (R - m) /s

        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)

        print("Iter:", t, "Avg Reward: %.3f" % m, "Max: %.3f" % R.max(), "Duration:", datetime.now() - t0)

    return params, reward_per_iteration
        


# In[8]:


def softmax(a):
    c = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - c)
    return e / e.sum(axis=-1, keepdims=True)


# In[9]:


def sortino_ratio(returns, target=0):
    downside = returns[returns < target]
    downside_deviation = np.sqrt(np.mean((downside - target)**2)) \
        if len(downside) > 0 else 1e-8
    return (np.mean(returns) - target) / downside_deviation
    


# In[10]:


def reward_function(params, returns=train_returns, plot=False):
    weights = softmax(params)
    portfolio_returns = returns @ weights # (T, D) * (D, 1)

    if plot:
        cumulative_gross_return = np.cumprod(portfolio_returns + 1)

        # benchmark returns
        benchmark_returns = returns @ benchmark_weights
        cumulative_benchmark_return = np.cumprod(benchmark_returns + 1)

        plt.plot(cumulative_benchmark_return, label='benchmark')
        plt.plot(cumulative_gross_return, label='portfolio')
        plt.legend()
        plt.title('Cumulative Gross Return')
        plt.show()

    return sortino_ratio(portfolio_returns)


# In[ ]:


if __name__ == '__main__':
    # thread pool for parallelization
    pool = Pool(4)

    # train and save
    params = np.random.randn(len(tickers) + 1)
    best_params, rewards = evolution_strategy(
        f=reward_function,
        population_size=30, # this gets > 2k
        sigma=0.05,
        lr=0.02,
        initial_params=params,
        num_iters=50,
        pool = pool
    )

    # plot the rewards per iteration
    plt.plot(rewards)
    plt.show()

    print("Best weights:")
    best_weights = softmax(best_params)
    print(best_weights)
    for w, t in zip(best_weights, tickers + ['Cash']):
        print(f"{t} {w:.3f}")
        

    # plot portfolio performance over time train
    sr = reward_function(best_params, returns=train_returns, plot=True)
    print("Sortino Ratio Train:", sr)
    
    # plot portfolio performance over time test
    sr = reward_function(best_params, returns=test_returns, plot=True)
    print("Sortino Ratio Test:", sr)

    # sortino ratio for all BTC
    all_btc = np.array([0, 0, 0, 1, 0])
    sr = reward_function(all_btc, returns=test_returns, plot=True)
    print("Sortino Ratio BTC:", sr)

    # sortino ratio for benchmark
    portfolio_returns = test_returns @ benchmark_weights
    sr = sortino_ratio(portfolio_returns)
    print("Sortino Ratio Benchmark:", sr)


# In[ ]:




