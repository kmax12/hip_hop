from topic_model import TopicModel
import json
from collections import defaultdict
import numpy as np 
import gensim
import heapq
from scipy.interpolate import spline
from sklearn.decomposition import PCA, KernelPCA
import inspect
import pdb
import tsne

def group_by(songs,keys,weighted=False, filter_func=lambda x:True):
    dist = defaultdict(list)

    #aggregate by keys
    for s in songs:
        if not filter_func(s):
            print 'skipped'
            continue

        if type(keys) == list:
            k = (s.get(k, None) for k in keys)
        else:
            k = s.get(keys, None)

        if weighted:
            dist[k].append(s['weighted_topic_dist']) 
        else:
            dist[k].append(s['topic_dist']) 

    return dist


def count_by(songs, keys,weighted=False,filter_func=lambda x:True):
    dist = group_by(songs, keys,weighted,filter_func=filter_func)

     #normalize
    for v in dist:
        dist[v] =   len(dist[v])

    return dist
   

def avg_by(songs, keys,weighted=False,filter_func=lambda x:True):
    dist = group_by(songs, keys,weighted,filter_func=filter_func)

     #normalize
    for v in dist:
        dist[v] =   [sum(x) for x in zip(*dist[v])]
        l = np.sum(dist[v])
        dist[v] = [t/l for t in dist[v]]

    return dist

def process_songs(songs):
    #process album metadata to get year data
    albums = json.load(open('album_meta.json', 'r'))['results']['metadata']
    album_years = {}
    for a in albums:
        year = a['year'][1:-1]
        if year != "":
            album_years[a['url']] = int(year)

    #process songs
    for i,s in enumerate(songs):
        #add year data to songs
        if 'album_url' in songs[i]:
            album_url = songs[i].get('album_url', None)
            songs[i]['year'] = album_years.get(album_url, None)

        if 'views' in songs[i]:
            val = songs[i]['views'][:-6].replace(',','')
            songs[i]['views'] = int(val)

    return songs


def make_topic_model(songs_texr, NUM_TOPICS=10,USE_MODEL_CACHE=False, MODEL_CACHE_NAME = 'topic_model_cache'):
    MODEL_CACHE_NAME += "_" + str(NUM_TOPICS)
    if USE_MODEL_CACHE:
        print 'Loading topic model from cache'
        model = TopicModel.load(MODEL_CACHE_NAME)
    else:
        model = TopicModel(num_topics=NUM_TOPICS)
        model.build(songs_text)
        model.save(MODEL_CACHE_NAME)

    return model

def year_histogram(songs):
    counts = count_by(songs, 'year', weighted)
    




def topics_by_year(songs, weighted=False):
    year_data = avg_by(songs, 'year', weighted)

    # print count_by(songs, 'year', weighted)[None]
    
    num_topics = len(songs[0]['topic_dist'])
    series = []
    for topic_num in range(num_topics):
        name = "Topic "+str(topic_num+1) 
        add = {
            'name' : name,
            'marker': {'enabled':False},
            'data': []
        }
        for year in year_data:
            if year != None:
                val = year_data[year][topic_num]*100
                add['data'].append([year, val])
        series.append(add)       

    file_name = "topic_year_unweighted.json"
    if weighted:
        file_name = "topic_year_weighted.json"
    with open(file_name, "w") as outfile:
        json.dump(series, outfile, indent=4)

    return series

def top_for_topic(topic_num, keys, model, songs, n=3):
    vals = avg_by(songs, keys)
    return heapq.nlargest(n, vals, key=lambda x: vals.get(x)[topic_num])

def topic_info(model, songs):
    num_topics = model.num_topics
    topic_to_words = {}
    for topic_num in range(num_topics):
        name = "Topic "+str(topic_num+1) 
        words = [x[0] for x in  model.words_for_topic(topic_num, 10)]
        topic_to_words[name] = {
            'words' : words,
            # 'top_songs' : map(list,top_for_topic(topic_num, ['title', 'url'], model, songs)),
            # 'top_artists' : top_for_topic(topic_num,  'artist', model, songs),
            # 'top_albums' : map(list,top_for_topic(topic_num,['album', 'album_url'], model, songs))
        }

    with open('topic_info.json', "w") as outfile:
        json.dump(topic_to_words, outfile, indent=4)

    return topic_to_words


def similarity(songs, keys, filter_func= lambda x: True):
    # pdb.set_trace()
    avgs = avg_by(songs, keys, filter_func=filter_func)
    print len(avgs)
    pca = PCA(n_components=2)
    vals = avgs.values()
    print vals
    # pca.fit(vals)
    Y = tsne.tsne(np.array(vals), 2, 50, 20.0);

    points = []
    for (k,c) in zip(avgs,Y):
        # c = pca.transform(avgs[k])[0].tolist()
        p = {
            'x' : c[0],
            'y' : c[1],
            'song_title': k.next(),
            'song_url' : k.next()
            # 'z' : c[2],
        }
        points.append(p)

    series = {
        'name' : keys,
        'data' : points
    }


    filename = "%s_similarities.json" % (keys)
    with open(filename, "w") as outfile:
        json.dump(series, outfile, indent=4)




#load in and process songs
songs = json.load(open('songs.json', 'r'))['results']['details']
songs = process_songs(songs)


#find topics
USE_MODEL_CACHE = True
songs_text = songs_text = [s['lyrics'] for s in songs] #grab just lyric text from songs
# model = make_topic_model(songs_text, 3, USE_MODEL_CACHE)
# model = make_topic_model(songs_text, 5, USE_MODEL_CACHE)
# model = make_topic_model(songs_text, 10, USE_MODEL_CACHE)
# model = make_topic_model(songs_text, 25, USE_MODEL_CACHE)
model = make_topic_model(songs_text, 50, USE_MODEL_CACHE)
# model = make_topic_model(songs_text, 100, USE_MODEL_CACHE)
# model = make_topic_model(songs_text, 200, USE_MODEL_CACHE)


topics = model.model_docs(songs_text)


#add topic distrubtions back to songs list
for i,t_dist in enumerate(topics):
    songs[i]['topic_dist'] = gensim.matutils.sparse2full(t_dist, model.num_topics)
    songs[i]['weighted_topic_dist'] = gensim.matutils.sparse2full(t_dist, model.num_topics)*songs[i]['views'] #unnormailed distribution weighted by song popularity


# similarity(songs, 'artist')
similarity(songs, ['title','url', 'artist'],  lambda s: s['views'] > 500000)
# topic_info(model, songs)
# topics_by_year(songs, weighted=False)
# topics_by_year(songs, weighted=True)