from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

m = Prophet()
df = pd.read_csv('laundry.csv')
df.columns = ['ds', 'y']
m.add_seasonality(
    name='weekly', period=7, fourier_order=3, prior_scale=0.1)
date_ = pd.date_range(start='1/1/2018', end='1/20/2020', freq='D')
print(date_)
df['ds'] = date_
future = pd.date_range(start='1/21/2020', end='2/10/2020')
future = pd.DataFrame({"ds": future})
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
fig.savefig('fb.jpg')
m.plot(forecast)
plt.show()



