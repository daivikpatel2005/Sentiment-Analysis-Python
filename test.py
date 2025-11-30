# test.py

from sentiment_engine import SentimentEngine

engine = SentimentEngine("model.pkl", "vectorizer.pkl")

text = input("Enter a review: ")

result = engine.predict(text)

print(result)
