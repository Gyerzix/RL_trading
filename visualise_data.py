import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

df = pd.read_pickle('data/binance-BTCUSDT-1h.pkl')
df = df.reset_index()

df = df.rename(columns={'date_open': 'date'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

df_ohlc = df[['date', 'open', 'high', 'low', 'close']].copy()
df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

candlestick_ohlc(
    ax1,
    df_ohlc.values,
    width=0.002,
    colorup='g',
    colordown='r',
    alpha=0.8
)
ax1.set_title('BTC/USDT Price (1h) - Binance')
ax1.set_ylabel('Price (USDT)')
ax1.xaxis_date()

ax2.bar(df['date'], df['volume'], color='blue', alpha=0.6)
ax2.set_ylabel('Volume')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

output_path = "BTC_USDT_1h_candles.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"График сохранен в файл: {output_path}")