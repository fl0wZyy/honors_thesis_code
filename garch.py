from arch import arch_model
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from arch.__future__ import reindexing

df = pd.read_csv('spy_data.csv')

returns = 100 * df.Close.pct_change().dropna()
dates = df['Date']
plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel('Percentage Return')
plt.title('SPY Returns')


plot_pacf(returns ** 2)
plt.show()
forecasts = {}
model = arch_model(returns, p=2, q=20)
model_fit = model.fit()

rolling_predictions = []
rolling_dates = []
test_size = 365*5
for i in range(test_size-20):
    train = returns[:-(test_size-i)]
    date = list(dates[:-(test_size-i)])[-1]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=20)
    values = pred.variance.values
    value_20_day = sum(values[0])
    value_20_day = np.sqrt(value_20_day)
    value_20_day = value_20_day/np.sqrt(20)
    rolling_predictions.append(value_20_day)
    rolling_dates.append(date)
output = pd.DataFrame(list(zip(rolling_dates,rolling_predictions)), columns=['Last Date','1-Mo Volatility'])

output.to_csv(r'garch_output.csv')

