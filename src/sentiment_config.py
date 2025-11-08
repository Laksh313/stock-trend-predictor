"""
Configuration for news sentiment analysis
"""

# NewsAPI Configuration
NEWS_API_CONFIG = {
    'api_key': '47b95efc72314aefafb77a57c3df468c',  # Get free key from https://newsapi.org/
    'base_url': 'https://newsapi.org/v2/everything',
    'language': 'en',
    'sort_by': 'publishedAt',  # Most recent first
}

# Sentiment Analysis Settings
SENTIMENT_CONFIG = {
    'enabled': True,              # Enable/disable sentiment analysis
    'news_days': 3,               # Look back N days for news
    'min_articles': 2,            # Minimum articles needed for analysis
    'max_articles': 20,           # Maximum articles to analyze
    'show_headlines': 3,          # Number of headlines to display
    'method': 'vader',            # 'vader' or 'finbert' (vader is faster)
}

# Sentiment Score Thresholds
SENTIMENT_THRESHOLDS = {
    'very_negative': -0.6,
    'negative': -0.3,
    'slightly_negative': -0.1,
    'neutral': 0.1,
    'slightly_positive': 0.3,
    'positive': 0.6,
    'very_positive': 1.0,
}

# Sentiment Labels and Icons
SENTIMENT_LABELS = {
    'very_negative': {'label': 'Very Negative', 'icon': '❌❌', 'color': 'red'},
    'negative': {'label': 'Negative', 'icon': '❌', 'color': 'red'},
    'slightly_negative': {'label': 'Slightly Negative', 'icon': '⚠️', 'color': 'yellow'},
    'neutral': {'label': 'Neutral', 'icon': '➖', 'color': 'gray'},
    'slightly_positive': {'label': 'Slightly Positive', 'icon': '✅', 'color': 'green'},
    'positive': {'label': 'Positive', 'icon': '✅✅', 'color': 'green'},
    'very_positive': {'label': 'Very Positive', 'icon': '✅✅✅', 'color': 'green'},
}

# Stock ticker to company name mapping (for better news search)
TICKER_TO_COMPANY = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google Alphabet',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'META': 'Meta Facebook',
    'NVDA': 'Nvidia',
    'AMD': 'AMD',
    'JPM': 'JPMorgan Chase',
    'BAC': 'Bank of America',
    'XOM': 'Exxon Mobil',
    'JNJ': 'Johnson Johnson',
    'WMT': 'Walmart',
    'PG': 'Procter Gamble',
    'V': 'Visa',
    'MA': 'Mastercard',
    'DIS': 'Disney',
    'NFLX': 'Netflix',
    'BA': 'Boeing',
    'KO': 'Coca Cola',
}

# Cache settings (to avoid hitting API limits)
CACHE_CONFIG = {
    'enabled': True,
    'cache_dir': 'data/sentiment_cache',
    'cache_duration_hours': 2,  # Cache news for 2 hours
}

# Agreement analysis thresholds
AGREEMENT_CONFIG = {
    'strong_agreement': 0.4,      # Both indicators strongly aligned
    'weak_agreement': 0.2,        # Both indicators weakly aligned
    'neutral_sentiment': 0.1,     # Sentiment is too neutral to matter
}

# Source credibility weights (optional - for advanced weighting)
SOURCE_WEIGHTS = {
    'Bloomberg': 1.5,
    'Reuters': 1.5,
    'CNBC': 1.3,
    'Wall Street Journal': 1.4,
    'Financial Times': 1.4,
    'MarketWatch': 1.2,
    'Yahoo Finance': 1.1,
    'Seeking Alpha': 1.0,
    'default': 1.0,
}

# Error messages
ERROR_MESSAGES = {
    'no_api_key': 'NewsAPI key not configured. Please add your API key to sentiment_config.py',
    'api_limit': 'NewsAPI rate limit reached. Please try again later or upgrade your plan.',
    'no_news': 'No recent news found for this stock.',
    'network_error': 'Could not fetch news. Please check your internet connection.',
}

# Display settings
DISPLAY_CONFIG = {
    'show_individual_scores': False,  # Show score for each article
    'show_source': True,              # Show article source
    'show_date': True,                # Show article date
    'show_url': False,                # Show article URL
    'max_headline_length': 80,        # Truncate long headlines
}