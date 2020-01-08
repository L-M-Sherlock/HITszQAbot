# encoding=utf-8
import numpy as np
import re
import pandas
import itertools
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from imblearn.over_sampling import RandomOverSampler

le = LabelEncoder()
oe = OneHotEncoder()
# ros = RandomOverSampler()
np.random.seed(4)


# model = KeyedVectors.load_word2vec_format('/home/ngly/datasets/glove/results/glove.6B.100d.txt', binary=True)

# model = Word2Vec.load('mnt/DialogueApi/WMD/data/zh_wiki_word2vec_model')
# model_dims = 400
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_from_disk(data_path):
    # Load dataset from file
    train_path = os.path.join(data_path, "train.txt")
    dataset = pandas.read_csv(train_path, encoding='utf-8', names=['comments', 'label'], sep='\t', header=None)
    # print dataset.shape
    # Split by words
    X = [clean_str(sentence) for sentence in dataset['comments']]
    X = [list(sentence) for sentence in X]
    Y = np.array(dataset['label'])
    # for _, i in enumerate(Y):
    #    if np.isnan(i):
    #        print _
    # X = np.array(X)
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)

    # = oe.fit_transform(Y).toarray()
    # X, Y = ros.fit_sample(X, Y)

    return [X, Y]


def inverse_label(prediction):
    return list(le.inverse_transform([prediction]))[0]


def load_dev():
    dataset = pandas.read_csv('./test.txt', encoding='utf-8', names=['comments', 'label'], sep='\t', header=None)
    # print dataset.shape
    X = [clean_str(sentence) for sentence in dataset['comments']]
    X = [list(sentence) for sentence in X]
    Y = np.array(dataset['label'])
    Y_set = set(Y)
    Y = le.fit_transform(Y)
    X = np.array(X)
    Y = Y.reshape(-1, 1)
    Y = oe.fit_transform(Y).toarray()
    # X, Y = ros.fit_sample(X, Y)
    return [X, Y]


def pad_sentences(sentences, padding_word="<PAD/>", maxlen=0):
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """

    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i][:maxlen]
        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        # sentence = ''.join(sentence)
        # sentence = list(jieba.cut(sentence))
        # num_padding = sequence_length - len(sentence)
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_word2vec(sentences, labels, maxlen=0):
    x = []
    for sentence in sentences:
        s = []
        for word in sentence:
            try:
                word_vec = np.array(model[word]).reshape((1, model_dims))
            except:
                word_vec = np.random.uniform(low=-0.1, high=0.1, size=(1, model_dims))
            s.append(word_vec)

        x.append(s)

    x = np.array(x)
    x = np.reshape(x, (-1, maxlen, model_dims))
    y = np.array(labels)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_data(date_path):
    sentences, labels = load_data_from_disk(date_path)
    # sentences_dev, labels_dev = load_dev()
    # print np.array(sentences).shape
    # print np.array(sentences_dev).shape
    sentences_padded = pad_sentences(sentences, maxlen=100)
    # sentences_padded_dev = pad_sentences(sentences_dev, maxlen=100)
    # print np.array(sentences_padded).shape
    # print np.array(sentences_padded_dev).shape
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y = build_input_data(sentences_padded, labels, vocabulary)
    # x_dev, y_dev = build_input_data(sentences_padded_dev, labels_dev, vocabulary)

    # x, y = ros.fit_sample(x,y)
    y = y.reshape((-1, 1))
    y = oe.fit_transform(y).toarray()
    # print x.shape, y.shape, x_dev.shape, y_dev.shape

    return [x, y, vocabulary, vocabulary_inv]


def load_data_word2vec():
    sentences_train, labels_train = load_data_from_disk()
    sentences_dev, labels_dev = load_dev()
    sentences_padded_train = pad_sentences(sentences_train, maxlen=100)
    sentences_padded_dev = pad_sentences(sentences_dev, maxlen=100)
    x_train, y_train = build_input_data_word2vec(sentences_padded_train, labels_train, maxlen=100)
    x_dev, y_dev = build_input_data_word2vec(sentences_padded_dev, labels_dev, maxlen=100)
    # print x_train[0], y_train[0]
    return [x_train, y_train, x_dev, y_dev]


if __name__ == '__main__':
    # load_data_from_disk()
    x, y, x_dev, y_dev, voc, voc_inv = load_data()

    print(x.shape, y.shape)
    print(x_dev.shape, y_dev.shape)
    # X, y = load_data_merge_cnn_rnn()
    # print X.shape, y.shape
    # x_train, y_train, x_dev, y_dev = load_data_word2vec()
    # print x_train.shape, y_train.shape, x_dev.shape, y_dev.shape
