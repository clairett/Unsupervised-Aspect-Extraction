from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
import codecs
import os
import csv

def parseSentence(line):
    pos_to_wornet_dict = {
        'JJ': wn.ADJ,
        'JJR': wn.ADJ,
        'JJS': wn.ADJ,
        'RB': wn.ADV,
        'RBR': wn.ADV,
        'RBS': wn.ADV,
        'NN': wn.NOUN,
        'NNP': wn.NOUN,
        'NNS':wn.NOUN,
        'NNPS': wn.NOUN,
        'VB': wn.VERB,
        'VBG': wn.VERB,
        'VBD': wn.VERB,
        'VBN': wn.VERB,
        'VBP': wn.VERB,
        'VBZ': wn.VERB,
    }
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop and i != ""]
    new_text = nltk.pos_tag(text_rmstop)
    text_stem = []
    for w in new_text:
        if w[1] in pos_to_wornet_dict:
            text_stem.append(lmtzr.lemmatize(w[0], pos=pos_to_wornet_dict[w[1]]))
        elif w[0].strip() != "":
            text_stem.append(w[0])
    return text_stem

def parse_sentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop and i != ""]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

def preprocess_csv(domain):
    # data preprocessing
    f = open('../dataset/'+domain+'.csv', 'r')
    if not os.path.exists('../preprocessed_data/' + domain):
        os.mkdir('../preprocessed_data/' + domain)
    out = codecs.open('../preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')
    sentences = set([])
    spamreader = csv.DictReader(f)
    count = 1
    for row in spamreader:
        if count > 2:
            if row['Q2'] != "":
                tokens = parse_sentence(row['Q2'])
                if len(tokens) > 0 and " ".join(tokens).strip() != "":
                    sentences.add(' '.join(tokens))
        count += 1
    out.write('\n'.join(list(sentences)))

def preprocess(domain):
    print('\t' + domain + ' set ...')
    preprocess_csv(domain)
    # print '\t'+domain+' train set ...'
    # preprocess_train(domain)
    # print '\t'+domain+' test set ...'
    # preprocess_test(domain)


print('Preprocessing raw review sentences ...')
# preprocess('restaurant')
# preprocess('beer')
preprocess('mobile')