import datetime as dt
import pandas_datareader as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt


st = dt.datetime(1990, 1, 1)
en = dt.datetime(2014, 1, 1)
data = pdr.stooq.StooqDailyReader(symbols="AMZN").read()
# returns = 100 * data["Adj Close"].pct_change().dropna()

std_volume = (data["Volume"] - data["Volume"].mean()) / data["Volume"].std()

model = sm.tsa.ARIMA(std_volume, order=(1, 0, 1)).fit()
print(model.summary())
data["Volume"].plot()
plt.show()
model.plot_diagnostics()
plt.show()
data["Volume"].plot.kde()
