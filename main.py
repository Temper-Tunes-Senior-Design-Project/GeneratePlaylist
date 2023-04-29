import requests
from flask import jsonify, Flask
from flask_cors import CORS, cross_origin
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from sklearn.linear_model import LogisticRegression 
import pickle
import numpy as np
import pandas as pd

import random
import warnings

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Entry point parameters: 
@app.route('/generatePlaylist')
@cross_origin()
def generatePlaylist(request):
    request_json = request.get_json(silent=True)
    if request_json and all(k in request_json for k in ("mood","percentage_new_songs","total_songs","closest_songs")):
        mood = request_json["mood"]
        percentage_new_songs = request_json["percentage_new_songs"]
        num_total_songs = request_json["total_songs"]
        closest_songs_list = request_json["closest_songs"]

    else:
        return ({"error":"Bad Input, must pass 'mood', 'percentage_new_songs', 'total_songs', and 'closest_songs'"}, 
                400)
    
    result = buildPlaylist(mood, percentage_new_songs, num_total_songs, closest_songs_list)
    if type(result) is str:
        return (jsonify({"error":result}), 503)
    return (jsonify({'songs': result}), 200)


#______________________________________________
# Initialization
#______________________________________________
#Setup Spotify and Firebase Credentials
sp = None
def spotify_client():
    global sp
    sp_cred = None
    with open('spotify_credentials.json') as credentials:
        sp_cred = json.load(credentials)
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(sp_cred["client_id"],sp_cred['client_secret']))


cred,db = None,None
def firestoreConnection():
    global cred
    global db
    cred = credentials.Certificate("mood-swing-6c9d0-firebase-adminsdk-9cm02-66f39cc0dd.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
LR_model = None
def load_LR_model():
    global LR_model
    with open('logreg_model1.pkl','rb') as f:
        LR_model = pickle.load(f)
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LogisticRegression was fitted with feature names")
#______________________________________________
# Playlist Generation
#______________________________________________
moods = ['sad','angry','energetic','excited','happy','content','calm','depressed'] #Represents DB indexing of moods

# Function to return new songs to be added to the user's generated playlist
# Takes the specified mood, the number of new songs, and the 5 closest songs to plug into the recommended function
def generateNewSongsList(mood, num_songs, closest_song_ids, old_songs_list): ###user_id???
    global sp
    if sp == None:
        spotify_client()
        firestoreConnection()
    
    mood_index = moods.index(mood) #Numbered label of mood
    #specify target and close moods
    acceptable_moods = [mood_index, (mood_index + len(moods) - 1) % len(moods), (mood_index + 1)%len(moods)] 
    song_count = 0
    count_mood_index = 0 #TESTING TO DETERMINE RATE OF SONG ADDITION
    count_other_moods = 0
    count_similar_moods = 0
    counts = {}
    
    # Use Spotipy to retrieve track information
    track_info = sp.tracks(closest_song_ids)['tracks']
    # Remove any elements that are None
    track_info = [track for track in track_info if track is not None]
    if len(track_info) == 0:
        return {"error": "could not identify any tracks in the closest songs list"}
    # Extract the URIs from the track information
    track_uris = [track['uri'] for track in track_info]
    final_song_ids = []
    while len(final_song_ids) < num_songs:        
        # Get recommended tracks and add them to final_song_ids if mood is classified as specified mood
        tracks = sp.recommendations(seed_tracks=track_uris[:5], limit=50)['tracks'] ##############################LIMIT
        track_ids = [track["id"] for track in tracks if track["id"] not in final_song_ids and track["id"] not in old_songs_list]
        #Split ids by whether they are already labelled or not
        known_track_moods_dict = getAlreadyLabelled(track_ids)
        new_track_ids = [track_id for track_id in track_ids if track_id not in known_track_moods_dict.keys()]
        # Get song features of the new ids
        features_df = retrieveTrackFeatures(new_track_ids)
        
        # Get predictions and update database
        # If there are no features or lyrics, and no songs are already labelled return an error
        if features_df.shape[0] == 0 and len(known_track_moods_dict.keys()) == 0:
            return {"error": "Issue with model and/or spotify server"}
        else:
            predictions = {}
            stop_loops = False

            for key, row in features_df.iterrows():
                if not(stop_loops):
                    data = row.values.reshape(1,-1)
                    LR_pred, LR_pred_probability = getMoodLabelLR(data)
                    predictions[key]=LR_pred
                    if LR_pred in acceptable_moods: song_count += 1
                    stop_loops = (song_count == num_songs)

            # Add song moods to DB
            addTrackMoodToDB(predictions)
            
            # Combine predictions and known labels
            # Currently, this will prioritize adding new songs to playlist over hits in our DB
            num_songs_remaining = num_songs - song_count
            ###STAT DICT FOR TESTING
            stat_dict = {**known_track_moods_dict, **predictions}
            #Shorten list of songs if necessary, start with the known moods (will add close moods as well)
            ids_to_add = [key for key in known_track_moods_dict.keys() 
                          if known_track_moods_dict[key] in acceptable_moods]
            if len(ids_to_add) > num_songs_remaining:
                ids_to_add = ids_to_add[:num_songs_remaining]
            song_count += len(ids_to_add)
        
        # Add the remaining (newly predicted) song IDs where mood = mood_index
        ids_to_add.extend([key for key in predictions.keys() if predictions[key] in acceptable_moods])
        final_song_ids.extend(ids_to_add)
        # TESTING TO DETERMINE RATE OF SONG ADDITION
        count_mood_index += len([key for key in stat_dict.keys() if stat_dict[key] == mood_index])
        count_other_moods += len(stat_dict) - len(ids_to_add)
        count_similar_moods += len([key for key in stat_dict.keys() 
                                    if stat_dict[key] != mood_index 
                                    and stat_dict[key] in acceptable_moods])
       
        ###TESTING
        print(f"count of specified moods: {count_mood_index}")
        print(f"count of other moods: {count_other_moods}")
        print(f"count of similar moods: {count_similar_moods}")
        
        unique_elements = range(8)
        for elem in unique_elements:
            count = list(stat_dict.values()).count(elem)
            counts[elem] = counts.get(elem,0) + count
        print(f"count of labels: {counts}")
        count_prob = {k: v/sum(counts.values()) for k,v in counts.items()}
        print(f"probability of labels: {count_prob}")


#           DB.update_user_liked_songs(UID,predictions.keys()) 
            # ^^Need to add a check to discard songs no longer on spotify, 
            # otherwise we might recommend songs that are no longer on spotify

#           DB.update_last_login(UID,datetime.utcnow())
    return final_song_ids

# Function that takes in the closest songs to the user's centroid and 
# Returns a list of randomly selected ids which favors the closest songs
# to the user's centroid
def generateOldSongsList(num_old_songs, closest_song_ids):
    if num_old_songs > len(closest_song_ids): return closest_song_ids
    song_ids_list = []
    final_list = []
    songs_to_iterate_over = closest_song_ids.copy()
    while len(final_list) < num_old_songs:
        song_ids_list = songs_to_iterate_over
        for index, song_id in enumerate(song_ids_list):
            # add an item to the list 70% of the time if we don't have enough songs yet
            if not(len(final_list) >= num_old_songs) and random.random() < 0.7:
                final_list.append(song_id)
                songs_to_iterate_over.pop(index)
    return final_list

# Combines the lists created by the generation functions to return the new list
def buildPlaylist(mood, percentage_new_songs, num_total_songs, closest_songs_list):
    # Generate a list of already liked songs
    # Check if there are enough songs in closest_songs_list
    num_old_songs = int(round((1 - percentage_new_songs) * num_total_songs))
    if len(closest_songs_list) < num_old_songs:
        num_old_songs = len(closest_songs_list)
        old_songs_list = closest_songs_list
    else:
        old_songs_list = generateOldSongsList(num_old_songs, closest_songs_list)
    # Generate a list of newer songs
    num_new_songs = num_total_songs - num_old_songs
    new_songs_list = generateNewSongsList(mood, num_new_songs, closest_songs_list[:5], old_songs_list)
    if type(new_songs_list) is dict:
        err = new_songs_list['error']
        return err
    
    # Combine the lists, shuffle and return the playlist (song ids)
    combined_ids = old_songs_list + new_songs_list 
    random.shuffle(combined_ids)
    return combined_ids

#______________________________________________
# Database Operations
#______________________________________________
            
def getTrackMoodFromDB(track_id):
    doc_ref = db.collection('songs').document(track_id)
    doc_data = doc_ref.get()
    if doc_data.exists:
        return doc_data.to_dict().get('mood')
    else:
        return None

def getAlreadyLabelled(track_ids):
    already_labelled = {}
    for track_id in track_ids:
        mood = getTrackMoodFromDB(track_id)
        if mood is not None:
            already_labelled[track_id] = mood
    return already_labelled

def addTrackMoodToDB(tracks_dict):
    for track_id, mood in tracks_dict.items():
        doc_ref = db.collection('songs').document(track_id)
        doc_ref.set({
            'mood': int(mood[0])
        })
    
#______________________________________________
# LR Model Classifcation
#______________________________________________

def getMoodLabelLR(songFeaturesDF):
    if LR_model is None:
        load_LR_model()
    prediction = LR_model.predict(songFeaturesDF)
    pred_probability= LR_model.predict_proba(songFeaturesDF)
    return prediction, pred_probability

def retrieveTrackFeatures(track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        current_features = sp.audio_features(track_ids[i:i+50])
        
        # Convert to DataFrame
        df = pd.DataFrame(current_features)
        
#         # Remove columns that we don't need
#         df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)
        
        df = df[['id', 'valence', 'energy']]
        
        # Append to list of dataframes
        dfs.append(df)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    features_df.set_index("id", inplace=True)
    #convert to dictionary, with track id as key
#     features_dict = features_df.set_index('id').T.to_dict('list')
    return features_df