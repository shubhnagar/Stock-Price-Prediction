import yfinance as yf
import pandas as pd  

ticker_symbol = "TATAMOTORS.NS"  
nifty50 = yf.Ticker(ticker_symbol)

intraday_data = nifty50.history(period="10y", interval="1d")
intraday_data.index = intraday_data.index.tz_localize(None)


if 'Dividends' in intraday_data.columns or 'Stock Splits' in intraday_data.columns:
    intraday_data = intraday_data[(intraday_data['Dividends'] == 0) & (intraday_data['Stock Splits'] == 0)]
    intraday_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')


def calculate_heikin_ashi(df):
    
    heikin_ashi = df.copy()
    heikin_ashi['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    heikin_ashi['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    heikin_ashi['HA_High'] = heikin_ashi[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    heikin_ashi['HA_Low'] = heikin_ashi[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    
    heikin_ashi.loc[0, 'HA_Open'] = df['Open'].iloc[0]

    
    heikin_ashi[['Original_High', 'Original_Low']] = df[['High', 'Low']]

    return heikin_ashi[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'Volume']]


heikin_ashi_data = calculate_heikin_ashi(intraday_data)


heikin_ashi_data['HA_Open'] = heikin_ashi_data['HA_Open'].fillna(heikin_ashi_data['HA_Close'].iloc[0])
heikin_ashi_data.ffill(inplace=True)

heikin_ashi_data.rename(
    columns={
        'HA_Open': 'Open',
        'HA_High': 'High',
        'HA_Low': 'Low',
        'HA_Close': 'Close',
        'Volume': 'Volume'
    },
    inplace=True
)

heikin_ashi_data.index = pd.to_datetime(heikin_ashi_data.index)


if heikin_ashi_data.index.tz is not None:
    heikin_ashi_data.index = heikin_ashi_data.index.tz_localize('UTC').tz_convert(None)

heikin_ashi_data.index = heikin_ashi_data.index.date


heikin_ashi_data.reset_index(inplace=True)


heikin_ashi_data.rename(columns={'index': 'Date'}, inplace=True)
heikin_ashi_data = heikin_ashi_data.iloc[:-1]


heikin_ashi_data.to_csv('dataset.csv', index=False)

print("Heikin-Ashi data has been saved to dataset.csv")