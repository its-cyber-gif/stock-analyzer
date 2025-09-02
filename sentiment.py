import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your NewsAPI key
API_KEY = "c376a7da26b4422bb1674e6bab985e57"

# Define last 30 days range
end_date = datetime.today().date()
start_date = end_date - timedelta(days=30)

# Query for "Apple"
url = (f"https://newsapi.org/v2/everything?"
       f"q=Apple&"
       f"from={start_date}&to={end_date}&"
       f"sortBy=publishedAt&"
       f"apiKey={API_KEY}")

response = requests.get(url).json()
articles = response.get("articles", [])

print(f"📰 Articles fetched: {len(articles)}")

# Convert to dataframe
# Convert to dataframe
if len(articles) > 0:
    news_df = pd.DataFrame([{
        "Date": art["publishedAt"][:10],
        "Title": art["title"]
    } for art in articles])

    # Handle missing titles
    news_df["Title"] = news_df["Title"].fillna("")

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    news_df["Sentiment"] = news_df["Title"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"] if x else 0
    )

    # Daily average sentiment
    daily_sentiment = news_df.groupby("Date")["Sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"Sentiment": "Daily_Sentiment"}, inplace=True)

    print("🔍 Sample headlines with sentiment:")
    print(news_df.head(5))
else:
    print("⚠️ No news articles found. Try different keywords or check API key.")
