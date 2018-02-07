import codecs
import re
import operator

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))

def create_vocab(domain, maxlen=0):
    # assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/'+domain+'/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {}
    top = 0

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue


        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab_size = 1500
    for _, freq in sorted_word_freqs:
        if freq >= 10:
            vocab_size += 1
    vocab_size = min(vocab_size, unique_words)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    #Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word+'\t'+str(0)+'\n')
            continue
        vocab_file.write(word+'\t'+str(word_freqs[word])+'\n')
    vocab_file.close()
    fin.close()
    return vocab


def determine_K_with_xmeans(data):
    print(len(data))
    print(data[0])

def determine_max_topics(domain):
    source = '../preprocessed_data/' + domain + '/train.txt'
    fin = codecs.open(source, 'r', 'utf-8')
    verbatim_count = len(fin.read().split('\n'))
    max_topics = 40
    if (verbatim_count >= 10000):
        max_topics = 50
    elif (verbatim_count > 8000):
        max_topics = 45
    elif (verbatim_count <= 8000):
        max_topics = 40
    return max_topics


def read_dataset(domain, phase, vocab, maxlen):
    # assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    source = '../preprocessed_data/'+domain+'/'+phase+'.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)
    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x



def get_data(domain, maxlen=0):
    print('Reading data from', domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, maxlen)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen)
    print('  test set')
    test_x, test_maxlen = read_dataset(domain, 'test', vocab, maxlen)
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen


if __name__ == "__main__":
    vocab, train_x, test_x, maxlen, aspect_size = get_data('fandango')
    # print(len(train_x))
    # print(len(test_x))
    # print(maxlen)

