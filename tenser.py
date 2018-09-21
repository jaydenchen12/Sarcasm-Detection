import csv
import os
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('sarcasm_v2.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        for (i,v) in enumerate(row):
            columns[i].append(v)

token = TweetTokenizer()
categories = [ "sarcastic", "regular" ]
filetype = columns[2]
text = columns[3]
words = []
docs = []
stemmer = LancasterStemmer()
for index in range(len(text)):
    try:
        if filetype[index].split("_")[1] == "sarc":
           #print (text[index])
           w = token.tokenize(text[index])
           words.extend(w)
           docs.append((w, "sarcastic"))
        else:
           w = token.tokenize(text[index])
           words.extend(w)
           docs.append((w, "regular"))
    except Exception:
        pass
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

training = []
output = []
output_empty = [0] * 2
#formatting training data
for doc in docs:
    bow = []
    token_words = doc[0]
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

def get_tf_record(sentence):
   global words
   w = token.tokenize(sentence)
   sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
   bow = [0]*len(words)
   for s in sentence_words:
      for i,w in enumerate(words):
         if w == s:
            bow[i] = 1
   return(np.array(bow))
print (model.predict([get_tf_record("the weather is nice")]))
