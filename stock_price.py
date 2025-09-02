import yfinance as yf
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Make sure VADER lexicon is downloaded
nltk.download("vader_lexicon")

# -------------------------------
# 1. Fetch Stock Data
# -------------------------------
ticker = "AAPL"
stock_df = yf.download(ticker, start="2025-08-05", end="2025-09-01")

# Flatten MultiIndex columns if present
if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_df.columns]

stock_df.reset_index(inplace=True)
stock_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

print("✅ Stock data fetched:", stock_df.shape)

# -------------------------------
# 2. Fetch News from NewsAPI
# -------------------------------
API_KEY = "c376a7da26b4422bb1674e6bab985e57"  # <-- Replace with your NewsAPI key
# Try searching for "Apple" instead of "AAPL"
query = "Apple"  
url = f"https://newsapi.org/v2/everything?q={query}&from=2022-01-01&to=2023-01-01&language=en&sortBy=publishedAt&apiKey={API_KEY}"

response = requests.get(url)
articles = response.json().get("articles", [])
print(f"📰 Articles fetched: {len(articles)}")
if len(articles) > 0:
    for i, art in enumerate(articles[:5]):
        print(f"{i+1}. {art['title']}")
# Create DataFrame
news_df = pd.DataFrame(articles)
if not news_df.empty:
    news_df["text"] = news_df["title"].fillna("") + " " + news_df["description"].fillna("")
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"]).dt.date
    news_df = news_df[["publishedAt", "text"]]
else:
    news_df = pd.DataFrame(columns=["publishedAt", "text"])

print("✅ News data fetched:", news_df.shape)

# -------------------------------
# 3. Sentiment Analysis with VADER
# -------------------------------
sia = SentimentIntensityAnalyzer()
if not news_df.empty:
    news_df["sentiment"] = news_df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    daily_sentiment = news_df.groupby("publishedAt")["sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"publishedAt": "Date", "sentiment": "Daily_Sentiment"}, inplace=True)
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])
else:
    daily_sentiment = pd.DataFrame(columns=["Date", "Daily_Sentiment"])

print("✅ Sentiment calculated:", daily_sentiment.shape)

# -------------------------------
# 4. Merge Stock Data with Sentiment
# -------------------------------
final_df = pd.merge(stock_df, daily_sentiment, on="Date", how="left")
final_df["Daily_Sentiment"] = final_df["Daily_Sentiment"].fillna(0)

# -------------------------------
# 5. Save Dataset
# -------------------------------
final_df.to_csv("stock_sentiment_dataset.csv", index=False)
print("💾 Dataset saved as stock_sentiment_dataset.csv")

# -------------------------------
# 6. Debug Output
# -------------------------------
print("\n🔍 Sample rows:")
print(final_df.head(10))
