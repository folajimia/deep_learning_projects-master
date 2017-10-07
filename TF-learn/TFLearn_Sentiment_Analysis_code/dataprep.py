import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter


# function to import
def data_import():

    reviews = pd.read_csv('reviews.txt', header=None)
    labels = pd.read_csv('labels.txt', header=None)
    return reviews, labels


def count_box(reviews):
    #Create the bag of words
    total_counts = Counter()
    for _, row in reviews.iterrows():
        total_counts.update(row[0].split(' '))
    print("Total words in data set: ", len(total_counts))

    #sort vocab by the count value and keep the 10000 most frequent words.
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
    print(vocab[:60])
    #Check the last word in the vocab to confirm that 1ast word  is not common
    print(vocab[-1], ': ', total_counts[vocab[-1]])

    # Dictionary called word2idx that maps each word in the vocabulary to an index
    word2idx = {word: i for i, word in enumerate(vocab)}

    return vocab, word2idx


#The function converts text to vector

def text_to_vector(text,vocab, word2idx):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)

#convert review data set to a word vector
def review_to_wordvector(reviews, vocab):
    word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
    for ii, (_, text) in enumerate(reviews.iterrows()):
        word_vectors[ii] = text_to_vector(text[0])
        return word_vectors



def split_data(reviews,labels,word_vectors):
    Y = (labels == 'positive').astype(np.int_)
    records = len(labels)

    shuffle = np.arange(records)
    np.random.shuffle(shuffle)
    test_fraction = 0.9

    train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):]
    trainX, trainY = word_vectors[train_split, :], to_categorical(Y.values[train_split, 0], 2)
    testX, testY = word_vectors[test_split, :], to_categorical(Y.values[test_split, 0], 2)

    return trainX, trainY,testX, testY



