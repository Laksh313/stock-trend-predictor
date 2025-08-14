import yfinance as yf
'''data = yf.Ticker("MSFT")
data.info
data.calendar
data.analyst_price_targets
data.quarterly_income_stmt
data.history(period='1mo')
data.option_chain(data.options[0]).calls'''

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock.to_csv(f"data/{ticker}_data.csv")
    print(f"Data saved to data/{ticker}_data.csv")

if __name__ == '__main__':
    fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')