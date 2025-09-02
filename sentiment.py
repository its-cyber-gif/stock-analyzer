import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------
# 1. Fetch Stock Data
# -------------------------------
ticker = "AAPL"
end_date = datetime.today().date()
start_date = end_date - timedelta(days=30)

stock_df = yf.download(ticker, start=start_date, end=end_date)

if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_df.columns]

stock_df.reset_index(inplace=True)
stock_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

stock_df.to_csv("stock_price.csv", index=False)
print(f"✅ Stock data saved ({stock_df.shape[0]} rows) → stock_price.csv")

# -------------------------------
# 2. Fetch News Articles
# -------------------------------
API_KEY = "c376a7da26b4422bb1674e6bab985e57"
url = (f"https://newsapi.org/v2/everything?"
       f"q=Apple&"
       f"from={start_date}&to={end_date}&"
       f"language=en&"
       f"sortBy=publishedAt&"
       f"apiKey={API_KEY}")

response = requests.get(url).json()
articles = response.get("articles", [])

print(f"📰 Articles fetched: {len(articles)}")

# -------------------------------
# 3. Sentiment Analysis
# -------------------------------
if len(articles) > 0:
    news_df = pd.DataFrame([{
        "Date": art["publishedAt"][:10],
        "Title": art.get("title", "")
    } for art in articles])

    analyzer = SentimentIntensityAnalyzer()
    news_df["Sentiment"] = news_df["Title"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"] if x else 0
    )

    daily_sentiment = news_df.groupby("Date")["Sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"Sentiment": "Daily_Sentiment"}, inplace=True)

    daily_sentiment.to_csv("stock_sentiment_dataset.csv", index=False)
    print(f"✅ Sentiment dataset saved ({daily_sentiment.shape[0]} rows) → stock_sentiment_dataset.csv")

else:
    print("⚠️ No news articles found. Try different keywords or check your API key.")

