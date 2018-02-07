import gensim
import codecs

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()

def train_and_save_local_embeddings(args):
    domain = args.domain
    print('Pre-training word embeddings ...')
    source = '../preprocessed_data/%s/train.txt' % (domain)
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=100, min_count=10, window=5, workers=4)
    model.save(model_file)
