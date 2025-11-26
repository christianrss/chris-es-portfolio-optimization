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

print(df.head())