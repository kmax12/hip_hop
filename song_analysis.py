from topic_model import TopicModel
import json
from collections import defaultdict
import numpy as np 
import gensim

def group_by(songs,key,weighted=False):
	dist = defaultdict(list)

	#aggregate by key
	for s in songs:
		k = s.get(key, None)
		if weighted:
			dist[k].append(s['weighted_topic_dist']) 
		else:
			dist[k].append(s['topic_dist']) 

	#normalize
	for v in dist:
		dist[v] =	[sum(x) for x in zip(*dist[v])]
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


def make_topic_model(songs_texr, USE_MODEL_CACHE=False, MODEL_CACHE_NAME = 'topic_model_cache'):
	
	if not USE_MODEL_CACHE:
		model = TopicModel()
		model.build(songs_text)
		model.save(MODEL_CACHE_NAME)
	else:
		print 'Loading topic model from cache'
		model = TopicModel.load(MODEL_CACHE_NAME)


	return model






#load in and process songs
songs = json.load(open('songs.json', 'r'))['results']['details']
songs = process_songs(songs)


#find topics
songs_text = songs_text = [s['lyrics'] for s in songs] #grab just lyric text from songs
model = make_topic_model(songs_text, True)
topics = model.model_docs(songs_text)


#add topic distrubtions back to songs list
for i,t_dist in enumerate(topics):
	songs[i]['topic_dist'] = gensim.matutils.sparse2full(t_dist, model.num_topics)
	songs[i]['weighted_topic_dist'] = gensim.matutils.sparse2full(t_dist, model.num_topics)*songs[i]['views'] #unnormailed distribution weighted by song popularity


print group_by(songs, 'year')