import requests
from flask import jsonify, Flask
from flask_cors import CORS, cross_origin
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


from sklearn.discriminant_analysis import StandardScaler
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import pickle
# #this folder location will be changed to DB location in actual cloud function
# from os import chdir
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bs4 import BeautifulSoup
from bs4 import element
from copy import deepcopy
import re
import random

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
    return (jsonify({'playlist': result}), 200)


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
    
RF_model = None
def load_RF_model():
    global RF_model
    with open('rf2.pkl','rb') as f:
        RF_model = pickle.load(f)

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
    song_count = 0
    count_mood_index = 0 #TESTING TO DETERMINE RATE OF SONG ADDITION
    count_other_moods = 0
    count_similar_moods = 0
    counts = {}
        
    #         #0. get the track ids of unlabelled songs from the list of new liked songs from the user
    #         last_login = DB.get_last_login(UID) #IF NONE, return ''

    #         #If last login is a string, we need to convert it to a utc datetime object
    #         #if isinstance(last_login, str):
    #         #        last_login = datetime.strptime(last_login, '%Y-%m-%d %H:%M:%S.%f')
    
    # Use Spotipy to retrieve track information
    track_info = sp.tracks(closest_song_ids)['tracks']
    # Remove any elements that are None
    track_info = [track for track in track_info if track is not None]
    # Extract the URIs from the track information
    if len(track_info) == 0:
        return {"error": "could not identify any tracks in the closest songs list"}
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
        # Preprocess features
        features_dict = clipAndNormalize(features_df)
        
        # Make a dictionary of song titles and artist names
        scraper_inputs = getTitlesAndArtists(new_track_ids)
        
        # Dictionary for lyrics of songs that are found
        all_lyrics_dict = getScrapedLyrics(scraper_inputs)
        
        overlap_keys = [key for key in features_dict.keys() if key in all_lyrics_dict.keys()]
        only_features = [key for key in features_dict.keys() if key not in overlap_keys]
        only_lyrics = [key for key in all_lyrics_dict.keys() if key not in overlap_keys]
        
        # Get predictions and update database
        # If there are no features or lyrics, and no songs are already labelled return an error
        if len(features_dict.keys()) == 0 and len(all_lyrics_dict.keys()) == 0 and len(known_track_moods_dict.keys()) == 0:
            #probably just return a flag that says there are no features or lyrics not the response here
            return {"error": "Issue with model and/or spotify server"}
        else:
            predictions = {}
            stop_loops = False
            
            #for first version, tokenize the lyrics and then pass then to the model inside the for loop
            for key in overlap_keys: #- could probably be done in batches regardless
                if not(stop_loops):
                    #################TEST
                    RF_pred, RF_pred_probability = getMoodLabelRF([features_dict[key]])
                    predictions[key]=RF_pred
                    if RF_pred == mood_index: song_count += 1
                    stop_loops = song_count == num_songs
                    ############
                    
#                     RF_pred, RF_pred_probability = getMoodLabelRF(features_dict[key])

#                     BERT_pred, RF_flag = getOnlyMoodLabelFromLyrics(all_lyrics_dict[key])
#                     if RF_pred == BERT_pred or RF_flag == True:
#                         prediction = RF_pred
#                     else:
#                         model_pred_diffs = (RF_pred - BERT_pred)
#                         if RF_pred > BERT_pred:
#                             sum_probabilities = RF_pred + model_pred_diffs
#                         else:
#                             sum_probabilities = RF_pred - model_pred_diffs
#                         #if sum_probabilities outside of below 0, then do 8-sum_probabilities
#                         if sum_probabilities < 0:
#                             prediction = 8 + sum_probabilities
#                         elif sum_probabilities > 7:
#                             prediction = sum_probabilities - 7
#                         else:
#                             prediction = sum_probabilities

#                     predictions[key]=prediction
#                     if prediction == mood_index: song_count += 1
#                     stop_loops = song_count == num_songs
                    

            for key in only_features:
                if not(stop_loops):
                    RF_pred, RF_pred_probability = getMoodLabelRF([features_dict[key]])
                    predictions[key]=RF_pred
                    if RF_pred == mood_index: song_count += 1
                    stop_loops = song_count == num_songs

            # Add song moods to DB
            #################### addTrackMoodToDB(predictions)
            
            # Combine predictions and known labels
            # Currently, this will prioritize adding new songs to playlist over hits in our DB
            num_songs_remaining = num_songs - song_count
            #############STAT DICT FOR TESTING
            stat_dict = {**known_track_moods_dict, **predictions}
            #Shorten list of songs if necessary, start with the known moods
            ids_to_add = [key for key in known_track_moods_dict.keys() if known_track_moods_dict[key] == mood_index]
            if len(ids_to_add) > num_songs_remaining:
                ids_to_add = ids_to_add[:num_songs_remaining]
            song_count += len(ids_to_add)
        
        # Add the remaining (newly predicted) song IDs where mood = mood_index
        ids_to_add.extend([key for key in predictions.keys() if predictions[key] == mood_index])
        final_song_ids.extend(ids_to_add) 
        # TESTING TO DETERMINE RATE OF SONG ADDITION
        count_mood_index += len([key for key in stat_dict.keys() if stat_dict[key] == mood_index])
        count_other_moods += len(stat_dict) - len(ids_to_add)
        count_similar_moods += len([key for key in stat_dict.keys() if 
                                   stat_dict[key] + len(moods) - 1 % len(moods) == mood_index or 
                                   stat_dict[key] + 1 % len(moods) == mood_index])
       
        ###TESTING
        print(f"count of specified moods: {count_mood_index}")
        print(f"count of other moods: {count_other_moods}")
        print(f"count of similar moods: {count_similar_moods}")
        
        unique_elements = set(range(8))
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
# RF Model Classifcation
#______________________________________________

def getMoodLabelRF(songFeatures):
    if RF_model is None:
        load_RF_model()
    prediction = RF_model.predict(songFeatures)
    pred_probability= RF_model.predict_proba(songFeatures)
    return prediction, pred_probability

def retrieveTrackFeatures(track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        current_features = sp.audio_features(track_ids[i:i+50])
        
        # Convert to DataFrame
        df = pd.DataFrame(current_features)
        
        # Remove columns that we don't need
        df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)
        
        
        # Append to list of dataframes
        dfs.append(df)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    
    
    #convert to dictionary, with track id as key
    #featuresDict = features_df.set_index('id').T.to_dict('list')
    
    return features_df


def clipAndNormalize(features):
    #clip the features to the range of the training data
    features['danceability'] = features['danceability'].clip(lower=0.25336000000000003, upper=0.9188199999999997)
    features['energy'] = features['energy'].clip(lower=0.047536, upper=0.982)
    features['loudness'] = features['loudness'].clip(lower=-24.65708, upper=-0.8038200000000288)
    features['speechiness'] = features['speechiness'].clip(lower=0.0263, upper=0.5018199999999997)
    features['acousticness'] = features['acousticness'].clip(lower=1.4072e-04, upper=0.986)
    features['instrumentalness'] = features['instrumentalness'].clip(lower=0.0, upper=0.951)
    features['liveness'] = features['liveness'].clip(lower=0.044836, upper=0.7224599999999991)
    features['valence'] = features['valence'].clip(lower=0.038318, upper=0.9348199999999998)
    features['tempo'] = features['tempo'].clip(lower=66.34576, upper=189.87784)
    features['duration_ms'] = features['duration_ms'].clip(lower=86120.0, upper=341848.79999999976)
    features['time_signature'] = features['time_signature'].clip(lower=3.0, upper=5.0)
    
    columns_to_log=['liveness', 'instrumentalness', 'acousticness', 'speechiness','loudness','energy']

    for i in columns_to_log:
        if i == 'loudness':
            features[i] = features[i] + 60
        features[i] = np.log(features[i]+1)

    #normalize the data
    scaler = pickle.load(open('scaler3.pkl', 'rb'))
    #fit on all columns except the track id
    rawfeatures = features.drop(['id'], axis=1)
    preprocessedFeatures = scaler.transform(rawfeatures)

    #convert to dictionary, with track id as key
    preprocessedFeatures = pd.DataFrame(preprocessedFeatures, columns=rawfeatures.columns)

    
    #apply z-score normalization
    for i in columns_to_log:
        preprocessedFeatures[i] = stats.zscore(preprocessedFeatures[i])
        preprocessedFeatures.clip(lower=-2.7, upper=2.7, inplace=True)

    preprocessedFeatures['id']= features['id']
    preprocessedFeatures = preprocessedFeatures.set_index('id').T.to_dict('list')
    return preprocessedFeatures


#______________________________________________
# Scraper Functions
#______________________________________________

def getTitlesAndArtists(track_ids):
    titleArtistPairs = {}
    for i in range(0,len(track_ids),50):
        tracks = sp.tracks(track_ids[i:i+50])
        for track in tracks['tracks']:
            title=track['name']
            #check if the track ends with (feat. artist) using a regex
            if re.search(r' \(feat. .*\)$', title):
                #remove the (feat. artist) from the title
                title = re.sub(r' \(feat. .*\)$', '', title)

            artists=[]
            for artist in track['artists']:
                artists.append(artist['name'])
            titleArtistPairs[track['id']] = {'title':title,'artist(s)':artists}

    return titleArtistPairs

def getScrapedLyrics(scraperInputs):
        all_lyrics_dict = {}
        for id, songInfo in scraperInputs.items():
                #maybe add a sleep or something to prevent getting blocked
                lyrics = scrapeLyrics(songInfo['artist(s)'],songInfo['title'])
                if len(lyrics) > 0:
                        all_lyrics_dict[id]=lyrics
        return all_lyrics_dict

#Helps parse miscellaneous tags <i>, </br>, etc,.
def _lyricsHelper(html, lyrics_list):
    for tag in html.childGenerator():
        if type(tag) == element.NavigableString:
            _handleLyricAppend(lyrics_list, tag.text.strip())
        elif tag.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
            lyrics_list.append('')
        elif html.name == 'a':
            _lyricsHelper(tag, lyrics_list)

#Reads the HTML for lyrics dividers (if they exist) and appends the lyrics line by line to a list
def _getLyricsFromHTML(html):
    lyrics = html.findAll("div", {"data-lyrics-container" : "true"})
    lyrics_list = ['']
    for segment in lyrics:
        for a in segment.childGenerator():
            lyric = None
            if type(a) == element.NavigableString:
                lyric = a.strip()
                _handleLyricAppend(lyrics_list, lyric)
            else:
                _lyricsHelper(a, lyrics_list)
            if a.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
                lyrics_list.append('')
    return lyrics_list

#Helper function to handle appending and manipulating lyrics_list. A new line is generated only for </br> tags
def _handleLyricAppend(lyrics_list, lyric):
    if lyric is not None:
        last_index = len(lyrics_list) - 1
        #Handle special cases (parenthesis and symbols stick with words for instance)
        if lyrics_list[last_index] != '' and (lyrics_list[last_index][-1] in ['(','[','{','<'] or lyric in [')',']','}','>','!','?',',','.']):
            lyrics_list[last_index] += lyric
        else:
            lyrics_list[last_index] += " " + lyric if lyrics_list[last_index] != '' else lyric

#Determines whether a song will need to be translated (returns the link if it does, otherwise returns None)
def _getSongTranslationLink(html):
    translation_tags = html.find_all('a', {"class": re.compile('TextButton*')})
    for tag in translation_tags:
        if "english-translations" in tag['href']:
            return tag['href']
    return None

#Determines whether a page exists
def _pageExists(html):
    return html.find('div', class_='render_404') == None
        
#function to scrape lyrics from genius, takes an array of artists, and songname
def scrapeLyrics(artistnames, songname):
    lyrics_list = []
    found = False
    i = 0
    html = None
    while i < len(artistnames) and not(found):
        artistname = artistnames[i]
        artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
        songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
        page_url = 'https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics'
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, 'html.parser') 
        found = _pageExists(html)
        i += 1
    if found:
        #check if there is an english translation
        translation_url = _getSongTranslationLink(html)
        if translation_url is not None:
            page = requests.get(translation_url)
            html = BeautifulSoup(page.text, 'html.parser') 
            lyrics_list = _getLyricsFromHTML(html)
        else:
            #If there isn't a translation, make sure it's in english in the first place
            english = False
            for script in html.findAll('script'):
                if "language\\\":\\\"en" in str(script):
                    english = True
            if english: lyrics_list = _getLyricsFromHTML(html)
    return lyrics_list


#______________________________________________
# BERT Sentiment Analysis Functions
#______________________________________________

def getOnlyMoodLabelFromLyrics(lyrics):
    
    #PART 1: DATA SETUP
    moods = ['sad','angry','energetic','excited','happy','content','calm','depressed']
    nums = [0, 1, 2, 3, 4, 5, 6, 7]

    # create dictionary mapping strings to integers
    mood_to_num = {mood: num for mood,num in zip(moods,nums)}
    
    #device = 'cuda' if cuda.is_available() else 'cpu'
    
    #change to ./path/goemotions_model
    BERT_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original",local_files_only=True)
    #change to ./path/goemotions_tokenizer
    BERT_Tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original",local_files_only=True)
    emotionsAsValenceArousal= { 'admiration':(.6,.4),'amusement':(.6,.2),'anger':(-.8,.6),'annoyance':(-.6,.6),'approval':(.8,.6),'caring':(.6,-.2),'confusion':(-.2,.2),'curiosity':(0,.4),'desire':(.6,.6),'despair':(-.8,-.6),'disappointment':(-.6,-.6),'disapproval':(-.8,.65),'disgust':(-.8,.2),'embarrassment':(-.6,.4),'envy':(-.6,.4),'excitement':(.6,.8),'fear':(-.6,.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'joy':(.8,.2),'love':(.8,.4),'nervousness':(-.4,.6),'optimism':(.6,.2),'pride':(.6,.1),'realization':(.2,.2),'relief':(.4,-.4),'remorse':(-.6,-.4),'sadness':(-.8,-.2),'surprise':(.2,.6),'neutral':(0,0)}

    emotion_dict = BERT_model.config.id2label


    #PART 2 - get the mood label
    mood,relyOnLinearModel = getMoodLabelFromLyrics(lyrics,BERT_model, BERT_Tokenizer, emotion_dict, emotionsAsValenceArousal, device='cpu',printValenceArousal=False)
    mood = mood_to_num[mood]
    return mood,relyOnLinearModel


def getMoodLabelFromLyrics(lyrics,model, tokenizer, emotion_dict, emotionsAsValenceArousal,printValenceArousal = False,disregardNeutral=True, printRawScores=False, printTopN=False,topScoresToPrint=3,max_length=512, device="cuda",  returnSongSegments=False):
    relyOnLinearResults = False
    softmaxScoresPerHeader = {}
    model.to(device)
    
    #part 1 - break up the lyrics into chunks and get the tokens
    if returnSongSegments:
        songTokenChunks,freqs,songSegs =breakUpSongByHeaders(lyrics,tokenizer,returnSongSegments=returnSongSegments,max_length=max_length, device=device)
    else:
        songTokenChunks,freqs =breakUpSongByHeaders(lyrics,tokenizer,returnSongSegments=returnSongSegments,max_length=max_length, device=device)

    #part 2 - get the softmax score for each chunk

    if len(songTokenChunks) == 1:
        disregardNeutral=False

    #softmax scores returns COMBINED SINGLE LABEL -- MAYBE TRY MULTIPLE LABELS AND TAKE THE MOST COMMON
    for header,tokenChunksPerHeaders in songTokenChunks.items():
        for tokenChunk in tokenChunksPerHeaders:
            ## ^^ If I encode multiple songs in batches, then I would make another for loop here and not just use tokenChunk[0]
            ## but it might be too complicated to do that this way.  
            # I'd have to make a function that breaks up the lyrics into chunks, 
            # and then return the chunks in a way that we still know which chunk belongs to which song and header
            if header not in softmaxScoresPerHeader:
                softmaxScoresPerHeader[header] = getSoftmax(model,tokenizer,tokens=tokenChunk[0],n=topScoresToPrint, printTopN=printTopN, printRawScores=printRawScores,device=device)
            else:
                softmaxScoresPerHeader[header] += getSoftmax(model,tokenizer,tokens=tokenChunk[0],n=topScoresToPrint, printTopN=printTopN, printRawScores=printRawScores,device=device)
            
            
    #Part 3 determine what to do with the neutral labels
    moodLabel, valence, arousal = convertScoresToLabels(softmaxScoresPerHeader,freqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral=disregardNeutral,printValenceArousal=printValenceArousal)

    if moodLabel=='top ratings all neutral':
        disregardNeutral=False
        moodLabel, valence, arousal = convertScoresToLabels(softmaxScoresPerHeader,freqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral=disregardNeutral,printValenceArousal=printValenceArousal)
        relyOnLinearResults = True
    if moodLabel=='neutral' or (-0.1<valence<0.1 and -0.1<arousal<0.1):
        relyOnLinearResults = True
    #part 4 - return the most common label
    return moodLabel, relyOnLinearResults


# input: a string of whole song
# output: a dictionary of with header values and a list of tensors (sometmes more than 1 item) for each header chunk
def breakUpSongByHeaders(songLines, tokenizer, max_length=512, device="cuda",  returnSongSegments=False):
    songSegmentsDict = {}
    tokenSegmentsDict = {}
    headerFreqsDict = {}

    #strip the trailing whitespace
    lines = [line.strip() for line in songLines]

    #find the lines that start with [ and end with ]
    headerLinesIndex = [i for i, line in enumerate(lines) if line.startswith('[') and line.endswith(']')]
    #check for consecutive headers indexes and remove the first one
    for i in range(len(headerLinesIndex)-1):
        if headerLinesIndex[i+1] - headerLinesIndex[i] == 1:
            headerLinesIndex[i] = -1
    headerLinesIndex = [i for i in headerLinesIndex if i != -1]

    for i in range(len(headerLinesIndex)):
        header_line = lines[headerLinesIndex[i]][1:-1]  # remove square brackets
        if header_line in songSegmentsDict:
            songSegmentsDict[header_line][0] += 1
        elif i == len(headerLinesIndex)-1:
            songSegmentsDict[header_line] = [1, " ".join(lines[headerLinesIndex[i]+1:]), lines[headerLinesIndex[i]+1:]]
        else:
            songSegmentsDict[header_line] = [1, " ".join(lines[headerLinesIndex[i]+1:headerLinesIndex[i+1]]), lines[headerLinesIndex[i]+1:headerLinesIndex[i+1]]]

    for header, lyrics in songSegmentsDict.items():
        if returnSongSegments:
            tokenSegmentsDict[header],subLyrics = breakUpLargeLyricChunks(lyrics[1],lyrics[2],tokenizer,returnLyricsSegments=returnSongSegments,max_length=max_length, device=device)
            songSegmentsDict[header]=subLyrics
        else:
            tokenSegmentsDict[header] = breakUpLargeLyricChunks(lyrics[1],lyrics[2],tokenizer,returnLyricsSegments=returnSongSegments,max_length=max_length, device=device)
        headerFreqsDict[header] = lyrics[0]

    if returnSongSegments:
        return tokenSegmentsDict,headerFreqsDict,songSegmentsDict
    else:
        return tokenSegmentsDict,headerFreqsDict




def breakUpLargeLyricChunks(lyricsChunkString, lines,tokenizer, max_length=512, device="cuda", returnLyricsSegments=False):
    #lines = lyricsChunkString.splitlines()  # split the lyrics into lines
    segments = []  # store the lyrics segments
    token_segments = []  # store the tokenized segments as tensors

    token_segment = tokenizer.encode(lyricsChunkString, return_tensors="pt")#.to(device)

    if len(token_segment[0]) <= max_length:
        token_segment = token_segment.unsqueeze(0)
        token_segments.append(token_segment)
        segments.append(lyricsChunkString)
    else:
        # calculate the average number of lines per segment. Add +2 to ensure segments are not still too long
        avg_lines_per_segment = len(lines) // ((len(token_segment[0]) // max_length) + 2)

        # loop through the lines and group them into segments of roughly the same length
        for start_idx in range(0, len(lines), avg_lines_per_segment):
            end_idx = start_idx + avg_lines_per_segment

            smallLastChunk = end_idx >= len(lines)-2
            
            if smallLastChunk:
                segment = " ".join(lines[start_idx:])
            else:
                segment = " ".join(lines[start_idx:end_idx])
            segments.append(segment)

            # tokenize the segment and convert to tensor
            token_segment = tokenizer.encode(segment, return_tensors="pt")#.to(device)
            token_segment = token_segment.unsqueeze(0)
            token_segments.append(token_segment)
            #NOTE: ^^ If I use batch_encode_plus, I can get the tokenized segments as a list of tensors in one step
            #I would just have to do it after the loop. 
            #Since it is a small list though, I don't think it will make a difference in this case

            if smallLastChunk:
                #this is the last segment early, so break out of the loop
                break

    if returnLyricsSegments:  
        return token_segments, segments
    else:
        return token_segments


def getSoftmax(model,tokenizer, tokens = None, sentence=None, n=3,printRawScores=False, printTopN=False,device='cuda'):
    if tokens is None:
        tokens = tokenizer.encode(sentence, return_tensors="pt")
    if device=='cuda':
        tokens = tokens.cuda()
    result = model(tokens)
    emotion = result.logits
    emotion = emotion.cpu().detach().numpy()
    emotion = emotion[0]
    softmax = tf.nn.softmax(emotion)
    #convert to numpy array
    softmax = softmax.numpy()
    if printRawScores:
        print(softmax)
    
    if printTopN:
        emotion = emotion.argsort()[-n:][::-1]
        emotion = emotion.tolist()
        printTopEmotions(emotion,model, softmax)
    return softmax

def printTopEmotions(emotion, model, softmax):
    
    #identify the label of top n emotions from emotion list
    #softmax is in the order of the values in emotion_dict so we can use emotion[id] to get the softmax value
    id=0
    emotion_dict = model.config.id2label
    for i in emotion:
        print(emotion_dict[i])
        print(softmax[emotion[id]]*100,"%")
        id+=1
    return


def convertScoresToLabels(softmaxScoresPerHeader,headerFreqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral = True, printValenceArousal=False,printTopChunkEmotions=False):
    #convert the softmax scores to a valence and arousal score
    #softmax scores are in the order of the values in emotion_dict so we can use emotion[id] to get the softmax value
    valence=0
    arousal=0
    softmaxScoresApplied=0
    #find the key in emotion_dict that corresponds to neutral
    neuturalKey = [key for key, value in emotion_dict.items() if value == 'neutral'][0]
    for key, softmaxScores in softmaxScoresPerHeader.items():
        #check if neutral is the highest softmax score
        if disregardNeutral and neuturalKey==softmaxScores.argmax():
            continue
        else:
            #multiply the softmax score by the valence and arousal values and add to the total valence and arousal
            #do this for the number in the headerFreqs dictionary
            for i in range(headerFreqs[key]):
                id=0
                softmaxScoresApplied+=1
                for i in softmaxScores:
                    valence+=i*emotionsAsValenceArousal[emotion_dict[id]][0]
                    arousal+=i*emotionsAsValenceArousal[emotion_dict[id]][1]
                    id+=1
    #divide the total valence and arousal by the number of softmax scores applied
    if softmaxScoresApplied!=0:
        valence=valence/softmaxScoresApplied
        arousal=arousal/softmaxScoresApplied
        mood =determineMoodLabel(valence,arousal,printValenceArousal=printValenceArousal)
        return mood, valence, arousal
    else:
        return 'top ratings all neutral', valence, arousal
    #note this means all top chunk emotions were neutral as opposed to true neutral where all emotions balance out to neutral

def determineMoodLabel(valence,arousal,printValenceArousal=False):
    #determine the diagonal of the circumplex model that the valence and arousal scores fall on
    #MAKE 2 BOXES OF THE CIRCUMPLEX MODEL A MOOD 

    energetic =   -0.5<valence<0.5 and arousal>0.5
    happy =       valence>0.5 and -.5<arousal<0.5
    calm =       -0.5<valence<0.5 and arousal<-0.5
    sad =         valence<-0.5 and -.5<arousal<0.5

    excited =   not (happy or energetic) and valence>0 and arousal>0
    content =   not (calm or happy) and valence>0 and arousal<0
    depressed = not (calm or sad) and valence<0 and arousal<0
    angry =   not (energetic or sad) and valence<0 and arousal>0


    if energetic:
        mood='energetic'
    elif happy:
        mood='happy'
    elif calm:
        mood='calm'
    elif sad:
        mood='sad'
    elif excited:
        mood='excited'
    elif content:
        mood='content'
    elif depressed:
        mood='depressed'
    elif angry:
        mood='angry'
    else:
        mood='neutral'
    
    if printValenceArousal:
        print("Valence: ",valence)
        print("Arousal: ",arousal)
    return mood     

if __name__ == '__main__':
    app = Flask(__name__)
    app.route('/generatePlaylist', methods=['POST'])(lambda request: generatePlaylist(request))
    app.run(debug=True)
