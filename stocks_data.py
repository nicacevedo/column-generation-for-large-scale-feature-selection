#%%
# Import yahoo finance data
# Import data from yahoo finance
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
import yfinance as yf

# Get the tickers from usa.csv
usa_tickers = pd.read_csv('usa.csv')['Symbol'].to_list()[:-2]
print(usa_tickers)

#%%

# Download data from yahoo finance by chunks of 500 tickers
start_year = 2010
data = []
for i in range(0, len(usa_tickers), 100):
    print(i)
    data.append(yf.download(
       ' '.join( usa_tickers[i:i+100]), 
        start=f'{start_year}-01-01', 
        end=datetime.date.today()
        )['Adj Close'])
    print(data[-1].shape)
    print(data[-1].head())

    # Concatenate data
    usa = pd.concat(data, axis=1)

# usa = yf.download(
#     usa_tickers, 
#     start=f'{start_year}-01-01', 
#     end=datetime.date.today()
#     )['Adj Close']
usa.head()
# Save usa data
usa.to_csv(f'real-data/usa_n{usa.shape[0]}_m{usa.shape[1]}_yr{start_year}.csv')

#%%

# Download S&P 500 (y variable) data from yahoo finance
# start_year = 2010
sp500 = yf.download(
    '^GSPC', 
    start=f'{start_year}-01-01', 
    end=datetime.date.today()
    )['Adj Close']
print(sp500.shape)
print(sp500.head())

sp500.round(4).to_csv(f'real-data/s&p_n{sp500.shape[0]}_yr{start_year}.csv')

# %%
