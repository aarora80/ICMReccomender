import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from textblob import TextBlob
import nltk
import ssl
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLTK's sentiment intensity analyzer
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Set up Spotify API credentials
client_id = '4afa4b079bee4d0a9dbde016173739ca'
client_secret = '2d111be07a7f40849f794b3a904c3883'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Musixmatch API credentials
musixmatch_api_key = '62ed208aed7571838c71787c85ed1bdb'

# Function to search for Bollywood songs on Spotify and retrieve their information
def search_bollywood_songs():
    results = sp.search(q='genre:"bollywood"', limit=10, type='track')  # Limiting to 10 songs for demonstration
    bollywood_songs = []
    for track in results['tracks']['items']:
        song_info = {
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'preview_url': track['preview_url'],
            'spotify_url': track['external_urls']['spotify']
            # Add more information as needed (e.g., track ID, album name)
        }
        # Get audio features, lyrics, and sentiment analysis for the track
        audio_features = get_audio_features(track['id'])
        lyrics = get_lyrics(track['name'], song_info['artist'])
        if audio_features:
            song_info['audio_features'] = audio_features
        if lyrics:
            song_info['lyrics'] = lyrics
            song_info['text_sentiment'] = analyze_lyrics_sentiment(lyrics)
        bollywood_songs.append(song_info)
    return bollywood_songs

# Function to get audio features for a track
def get_audio_features(track_id):
    audio_features = sp.audio_features(track_id)
    if audio_features:
        return audio_features[0]
    else:
        return None

# Function to get lyrics using Musixmatch API
def get_lyrics(track_name, artist_name):
    url = f'https://api.musixmatch.com/ws/1.1/matcher.lyrics.get?format=json&apikey={musixmatch_api_key}&q_track={track_name}&q_artist={artist_name}'
    response = requests.get(url)
    if response.status_code == 200:
        lyrics_data = response.json()
        if 'message' in lyrics_data and 'body' in lyrics_data['message'] and 'lyrics' in lyrics_data['message']['body']:
            return lyrics_data['message']['body']['lyrics']['lyrics_body']
    return None

# Function to analyze sentiment of lyrics using TextBlob
def analyze_lyrics_sentiment(lyrics):
    blob = TextBlob(lyrics)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Main function to gather the dataset and perform sentiment analysis
def perform_sentiment_analysis():
    bollywood_songs = search_bollywood_songs()
    print("Bollywood Songs:")
    for i, song in enumerate(bollywood_songs, 1):
        print(f"{i}. {song['name']} by {song['artist']} - Preview URL: {song['preview_url']}")
        if 'audio_features' in song:
            print("Audio Features:", song['audio_features'])
        if 'lyrics' in song:
            print("Lyrics:", song['lyrics'])
        if 'text_sentiment' in song:
            print("Text Sentiment:", song['text_sentiment'])

if __name__ == "__main__":
    perform_sentiment_analysis()
