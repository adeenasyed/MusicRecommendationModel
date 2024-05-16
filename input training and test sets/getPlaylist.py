import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

client_id = '2e9237962b9d488e897fbd9573d2c5b0'
client_secret = 'ae3069fe92844c82bb93480ce3ded7eb'

playlist_id = '7eHSuxOsJ67N4FUprxOrXx'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def get_track_features(track_id):
    features = sp.audio_features([track_id])[0]
    return features

def write_tracks_to_csv(tracks, filename='yasmins_playlist.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])
        
        for track in tracks:
            track_id = track['track']['id']
            features = get_track_features(track_id)
            if features:
                writer.writerow([features['danceability'], features['energy'], features['key'], features['loudness'], features['mode'], features['speechiness'], features['acousticness'], features['instrumentalness'], features['liveness'], features['valence'], features['tempo'], features['type'], features['id'], features['uri'], features['track_href'], features['analysis_url'], features['duration_ms'], features['time_signature']])

tracks = get_playlist_tracks(playlist_id)

write_tracks_to_csv(tracks)