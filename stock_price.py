import yfinance as yf
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")
ticker = "AAPL"
stock_df = yf.download(ticker, start="2023-01-01", end="2023-03-01")
if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_df.columns]
stock_df.reset_index(inplace=True)
stock_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
print(stock_df)
