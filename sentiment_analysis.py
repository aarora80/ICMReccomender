from textblob import TextBlob

def analyze_lyrics_sentiment(lyrics):
    """
    Analyzes the sentiment of the song lyrics and returns the emotional category.
    """
    blob = TextBlob(lyrics)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

def main():
    # Sample usage
    lyrics = """Sample lyrics go here"""
    emotion = analyze_lyrics_sentiment(lyrics)
    print("Emotion:", emotion)

if __name__ == "__main__":
    main()
