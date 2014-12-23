from sklearn.feature_extraction.text import CountVectorizer
import json
import cPickle as pickle
import gensim
import logging
from collections import defaultdict
import numpy as np 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class TopicModel():
    def __init__(self, num_topics=10, passes=10):
        self.num_topics = num_topics
        self.passes = passes

    def save(self, file_name):
        with open(file_name, "w") as out:
            pickle.dump(self, out)
    
    @staticmethod
    def load(file_name):
        with open(file_name, "r") as model_in:
            return pickle.load(model_in)


    def build(self, text_corpus):
        self.vectorizer = self.make_vectorizer(text_corpus)
        self.model = self.train_topic_model(text_corpus)

    def make_vectorizer(self, t):
        v = CountVectorizer(stop_words='english', min_df=.01, max_df=.5)
        v.fit(t)
        return v

    def log_perplexity(self,t):
        corpus, id2word = self.make_corpus(t)
        return self.model.log_perplexity(corpus)

    def train_topic_model(self, text_corpus, num_topics=None, passes=None):        
        if not num_topics:
            num_topics = self.num_topics

        if not passes:
            passes = self.passes

        if not self.vectorizer:
            raise Exception("must have vectorizer in order to train model")

        corpus, id2word = self.make_corpus(text_corpus)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, update_every=1, chunksize=1000, passes=passes)
        return model

    def make_corpus(self, t, v=None):
        """
        From a set of text documents, make a corpus to be used by gensim for topic modeling
        """
        v = self.vectorizer

        try:
            corpus = v.transform(t)
        except ValueError, e:
            return None, None
        
        vocab = {y:x for x,y in v.vocabulary_.iteritems()}
        corpus = gensim.matutils.Sparse2Corpus(corpus, documents_columns=False)
        return corpus, vocab

    def model_docs(self, text):
        if not self.model:
            raise Exception("must train model first")


        doc, vocab = self.make_corpus(text)
        
        if doc == None or doc == []:
            return None

        return self.model[doc] #only need first document

    def get_topic_distribution(self, docs):
        topic_dist = [0]*self.num_topics

        if len(docs) == 0:
            return topic_dist

        # print "start model"
        thread_topics  = self.model_docs(docs)

        # print "end modeling"
        for topics in thread_topics:
            if topics != None:
                for t_i, w_i in topics:
                    topic_dist[t_i] += w_i
        s = sum(topic_dist)
        
        if s ==0:
            return [0]*self.num_topics


        #remove small topics
        # for i, v in enumerate(topic_dist):
        #     v = v/s
        #     if v < 1.0/self.num_topics:
        #         topic_dist[i] = 0
        #     else:
        #         topic_dist[i] = v

        #renormalize
        s = sum(topic_dist)
        topic_dist = [i/s for i in topic_dist]

        return topic_dist

    def get_topic_num_hist(self, docs):
        topic_num_hist = [0]*(self.num_topics+1)

        if len(docs) == 0:
            return topic_dist

        # print "start model"
        thread_topics  = self.model_docs(docs)

        # print "end modeling"
        for topics in thread_topics:
            count = 0
            if topics != None:
                num_topics = len([w_i for (t_i,w_i) in topics if w_i > 0])
                topic_num_hist[num_topics] += 1

        return topic_num_hist

    def words_for_topic(self, topic_num, num_words):
        return [ (x[1].encode('utf-8'), x[0]) for x in self.model.show_topic(topic_num, num_words) ]


    def get_topics(self):
        return [Topic(self, i) for i in range(self.num_topics)]




def group_by(songs,key):
	dist = defaultdict(list)

	#aggregate by key
	for s in songs:
		k = s.get(key, None)
		dist[k].append(s['topic_dist']) 

	#normalize
	for v in dist:
		dist[v] =	[sum(x) for x in zip(*dist[v])]
		l = np.sum(dist[v])
		dist[v] = [t/l for t in dist[v]]


	return dist


if __name__ == "__main__":
		USE_MODEL_CACHE = True
		MODEL_CACHE_NAME = 'topic_model_cache'

		#load in songs
		songs = json.load(open('songs.json', 'r'))['results']['details']

		#process album metadata to get year data
		albums = json.load(open('album_meta.json', 'r'))['results']['metadata']
		album_years = {}
		for a in albums:
			year = a['year'][1:-1]
			if year != "":
				album_years[a['url']] = int(year)
		#add year data to songs
		for i,s in enumerate(songs):
			if 'album_url' in songs[i]:
				album_url = songs[i]['album_url']
				songs[i]['year'] = album_years.get(album_url, None)



		#grab just lyric text from songs
		songs_text = [s['lyrics'] for s in songs]


		if not USE_MODEL_CACHE:
			model = TopicModel()
			model.build(songs_text)
			model.save(MODEL_CACHE_NAME)
		else:
			print 'Loading topic model from cache'
			model = TopicModel.load(MODEL_CACHE_NAME)


		topics = model.model_docs(songs_text)


		#add topic distrubtions back to songs list
		for i,t_dist in enumerate(topics):
			songs[i]['topic_dist'] = gensim.matutils.sparse2full(t_dist, model.num_topics)




		print group_by(songs, 'year')