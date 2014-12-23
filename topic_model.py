from sklearn.feature_extraction.text import CountVectorizer
import cPickle as pickle
import gensim
import logging


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
