"""
News sentiment analysis for stock predictions
"""

import requests
import json
import os
from datetime import datetime, timedelta
from sentiment_config import (
    NEWS_API_CONFIG, SENTIMENT_CONFIG, SENTIMENT_THRESHOLDS,
    SENTIMENT_LABELS, TICKER_TO_COMPANY, CACHE_CONFIG,
    AGREEMENT_CONFIG, ERROR_MESSAGES, SOURCE_WEIGHTS
)

# Sentiment analyzers
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  VADER not installed. Install with: pip install vaderSentiment")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("⚠️  FinBERT not installed. Install with: pip install transformers torch")


class NewsSentimentAnalyzer:
    """Analyze news sentiment for stock predictions"""
    
    def __init__(self):
        self.api_key = NEWS_API_CONFIG['api_key']
        self.method = SENTIMENT_CONFIG['method']
        
        # Initialize sentiment analyzer
        if self.method == 'finbert' and FINBERT_AVAILABLE:
            print("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            print("✓ FinBERT loaded")
        elif VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            self.method = 'vader'
        else:
            raise ImportError("No sentiment analyzer available. Install vaderSentiment or transformers.")
    
    def get_company_name(self, ticker):
        """Get company name from ticker for better search"""
        return TICKER_TO_COMPANY.get(ticker.upper(), ticker)
    
    def fetch_news(self, ticker, days=None):
        """Fetch news articles for a stock ticker"""
        if days is None:
            days = SENTIMENT_CONFIG['news_days']
        
        # Check API key
        if self.api_key == 'YOUR_NEWSAPI_KEY_HERE':
            print(f"⚠️  {ERROR_MESSAGES['no_api_key']}")
            print("Get free API key from: https://newsapi.org/register")
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get company name for better search
        company_name = self.get_company_name(ticker)
        query = f"{ticker} OR {company_name}"
        
        # API parameters
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': NEWS_API_CONFIG['language'],
            'sortBy': NEWS_API_CONFIG['sort_by'],
            'apiKey': self.api_key,
            'pageSize': SENTIMENT_CONFIG['max_articles']
        }
        
        try:
            response = requests.get(NEWS_API_CONFIG['base_url'], params=params, timeout=10)
            
            if response.status_code == 429:
                print(f"⚠️  {ERROR_MESSAGES['api_limit']}")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                print(f"⚠️  API Error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            
            if not articles:
                print(f"⚠️  {ERROR_MESSAGES['no_news']}")
                return []
            
            return articles
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️  {ERROR_MESSAGES['network_error']}")
            print(f"Error details: {e}")
            return []
    
    def analyze_text_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        return scores['compound']  # Returns -1 to +1
    
    def analyze_text_finbert(self, text):
        """Analyze sentiment using FinBERT"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT outputs: [positive, negative, neutral]
        positive = predictions[0][0].item()
        negative = predictions[0][1].item()
        neutral = predictions[0][2].item()
        
        # Convert to -1 to +1 scale
        score = positive - negative
        return score
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        if self.method == 'finbert' and FINBERT_AVAILABLE:
            return self.analyze_text_finbert(text)
        else:
            return self.analyze_text_vader(text)
    
    def get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score <= SENTIMENT_THRESHOLDS['very_negative']:
            return 'very_negative'
        elif score <= SENTIMENT_THRESHOLDS['negative']:
            return 'negative'
        elif score <= SENTIMENT_THRESHOLDS['slightly_negative']:
            return 'slightly_negative'
        elif score <= SENTIMENT_THRESHOLDS['neutral']:
            return 'neutral'
        elif score <= SENTIMENT_THRESHOLDS['slightly_positive']:
            return 'slightly_positive'
        elif score <= SENTIMENT_THRESHOLDS['positive']:
            return 'positive'
        else:
            return 'very_positive'
    
    def analyze_articles(self, articles):
        """Analyze sentiment for multiple articles"""
        if not articles:
            return []
        
        analyzed = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', {}).get('name', 'Unknown')
            date = article.get('publishedAt', '')
            url = article.get('url', '')
            
            # Combine title and description for analysis
            text = f"{title}. {description}"
            
            # Analyze sentiment
            score = self.analyze_sentiment(text)
            
            # Apply source weight (optional)
            weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS['default'])
            weighted_score = score * weight
            
            analyzed.append({
                'title': title,
                'description': description,
                'source': source,
                'date': date,
                'url': url,
                'sentiment_score': score,
                'weighted_score': weighted_score,
                'sentiment_label': self.get_sentiment_label(score)
            })
        
        return analyzed
    
    def calculate_aggregate_sentiment(self, analyzed_articles):
        """Calculate overall sentiment metrics"""
        if not analyzed_articles:
            return None
        
        scores = [a['sentiment_score'] for a in analyzed_articles]
        weighted_scores = [a['weighted_score'] for a in analyzed_articles]
        
        # Overall sentiment (average)
        overall_sentiment = sum(weighted_scores) / len(weighted_scores)
        
        # Sentiment trend (recent vs older)
        if len(analyzed_articles) >= 4:
            recent_scores = scores[:len(scores)//2]
            older_scores = scores[len(scores)//2:]
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            trend_change = recent_avg - older_avg
            
            if trend_change > 0.1:
                trend = "Improving"
            elif trend_change < -0.1:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        # Find most positive and negative articles
        sorted_articles = sorted(analyzed_articles, key=lambda x: x['sentiment_score'], reverse=True)
        most_positive = sorted_articles[0] if sorted_articles else None
        most_negative = sorted_articles[-1] if sorted_articles else None
        
        # Get sentiment label
        sentiment_label_key = self.get_sentiment_label(overall_sentiment)
        sentiment_info = SENTIMENT_LABELS[sentiment_label_key]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_info['label'],
            'sentiment_icon': sentiment_info['icon'],
            'news_count': len(analyzed_articles),
            'sentiment_trend': trend,
            'most_positive': most_positive,
            'most_negative': most_negative,
            'all_articles': analyzed_articles[:SENTIMENT_CONFIG['show_headlines']]
        }
    
    def analyze_agreement(self, model_prediction, model_confidence, sentiment_score):
        """Analyze agreement between model and sentiment"""
        # Normalize model confidence to direction
        if model_prediction == 1:  # UP
            model_signal = model_confidence
        else:  # DOWN
            model_signal = -model_confidence
        
        # Check if sentiment is neutral
        if abs(sentiment_score) < AGREEMENT_CONFIG['neutral_sentiment']:
            return {
                'type': 'neutral_sentiment',
                'message': '📊 Rely on technical indicators - News provides no clear signal',
                'confidence': 'moderate'
            }
        
        # Check agreement
        if (model_signal > 0 and sentiment_score > 0) or (model_signal < 0 and sentiment_score < 0):
            # Both agree
            if abs(sentiment_score) > AGREEMENT_CONFIG['strong_agreement']:
                return {
                    'type': 'strong_agreement',
                    'message': '✅✅ Strong signal - Both technical indicators and news agree',
                    'confidence': 'high'
                }
            else:
                return {
                    'type': 'weak_agreement',
                    'message': '✅ Moderate signal - Technical and news sentiment align',
                    'confidence': 'moderate'
                }
        else:
            # Disagreement
            if model_prediction == 1:
                return {
                    'type': 'disagreement',
                    'message': '⚠️ Mixed signals - Model predicts UP but news sentiment is negative. Trade with caution.',
                    'confidence': 'low'
                }
            else:
                return {
                    'type': 'disagreement',
                    'message': '⚠️ Mixed signals - Model predicts DOWN but news sentiment is positive. Uncertain outlook.',
                    'confidence': 'low'
                }


def get_stock_sentiment(ticker, days=None):
    """
    Main function to get sentiment analysis for a stock
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to look back for news
    
    Returns:
        Dictionary with sentiment analysis results
    """
    if not SENTIMENT_CONFIG['enabled']:
        return None
    
    try:
        analyzer = NewsSentimentAnalyzer()
        
        print(f"\n📰 Fetching news for {ticker}...")
        articles = analyzer.fetch_news(ticker, days)
        
        if not articles:
            return {
                'available': False,
                'message': 'No recent news found'
            }
        
        if len(articles) < SENTIMENT_CONFIG['min_articles']:
            return {
                'available': False,
                'message': f'Only {len(articles)} articles found (minimum {SENTIMENT_CONFIG["min_articles"]} required)'
            }
        
        print(f"Analyzing sentiment for {len(articles)} articles...")
        analyzed = analyzer.analyze_articles(articles)
        sentiment_data = analyzer.calculate_aggregate_sentiment(analyzed)
        
        sentiment_data['available'] = True
        return sentiment_data
    
    except Exception as e:
        print(f"⚠️  Error in sentiment analysis: {e}")
        return {
            'available': False,
            'message': f'Error: {str(e)}'
        }


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"Testing sentiment analysis for {ticker}...")
        result = get_stock_sentiment(ticker)
        
        if result and result.get('available'):
            print(f"\nOverall Sentiment: {result['sentiment_label']} {result['sentiment_icon']}")
            print(f"Score: {result['overall_sentiment']:.2f}")
            print(f"Articles: {result['news_count']}")
            print(f"Trend: {result['sentiment_trend']}")
            
            if result['most_positive']:
                print(f"\nMost Positive: {result['most_positive']['title']}")
            if result['most_negative']:
                print(f"Most Negative: {result['most_negative']['title']}")
        else:
            print(f"\n{result.get('message', 'No sentiment data available')}")
    else:
        print("Usage: python news_sentiment.py AAPL")