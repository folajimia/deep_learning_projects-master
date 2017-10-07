import dataprep
import networkmodel
import pandas as pd
import numpy as np




#import data
reviews, labels = dataprep.data_import()

#get word and index
vocab, word2idx =dataprep.count_box(reviews)

dataprep.text_to_vector('The tea is for a party to celebrate the movie so she has no time for a cake'[:65], vocab, word2idx)


#text to vector function
def text_to_vector(text,vocab, word2idx):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)


#test function
def trialfn(testX,testY):
    predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int_)
    test_accuracy = np.mean(predictions == testY[:, 0], axis=0)
    print("Test accuracy: ", test_accuracy)



# Helper function that uses  model to predict sentiment
def try_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower(), vocab, word2idx)])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')




#convert review data set to a word vector
#def review_to_wordvector(reviews, vocab):
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0], vocab, word2idx)



trainX, trainY,testX, testY = dataprep.split_data(reviews,labels,word_vectors)


#Intializing the model
model = networkmodel.build_model()

#Training the network
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)

#test the network
trialfn(testX, testY)



sentence = "Moonlight is by far the best movie of 2016."
try_sentence(sentence)
#test
sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
try_sentence(sentence)



