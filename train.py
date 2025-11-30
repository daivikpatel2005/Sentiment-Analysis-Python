# train.py

from sentiment_engine import SentimentEngine

engine = SentimentEngine()
engine.train("dataset.csv")
