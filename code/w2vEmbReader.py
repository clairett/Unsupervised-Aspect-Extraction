import codecs
import logging
import numpy as np
import gensim
from sklearn.cluster import KMeans
import shelve


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class W2VEmbReader:

    def __init__(self, emb_path, vocab, emb_dim=None):

        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        # load google embeddings
        model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
        self.emb_dim = model.__dict__['vector_size']
        for word, index in vocab.items():
            if word in model.vocab:
                self.embeddings[word] = list(model.word_vec(word))
                emb_matrix.append(list(model.word_vec(word)))
            else:
                vector = np.random.uniform(-0.25, 0.25, self.emb_dim).astype('float32')
                self.embeddings[word] = list(vector)
                emb_matrix.append(list(vector))

        # # load local embeddings
        # model = gensim.models.Word2Vec.load(emb_path)
        # google = gensim.models.KeyedVectors.load_word2vec_format("~/GoogleNews-vectors-negative300.bin", binary=True)
        # # wiki_vector = shelve.open('../preprocessed_data/wiki.shelve', flag='r')
        # self.emb_dim = emb_dim
        # for word, index in vocab.items():
        #     if word in google.vocab:
        #         if word in model.wv.vocab:
        #             wv = list(np.array(np.concatenate((google.word_vec(word), model[word]))))
        #         else:
        #             wv = list(np.array(np.concatenate((google.word_vec(word), np.random.uniform(-0.25, 0.25, self.emb_dim-300).astype('float32')))))
        #         self.embeddings[word] = wv
        #         emb_matrix.append(wv)
        #     else:
        #         vector = np.random.uniform(-0.25, 0.25, self.emb_dim).astype('float32')
        #         self.embeddings[word] = list(vector)
        #         emb_matrix.append(list(vector))

        if emb_dim != None:
            assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix


    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim
