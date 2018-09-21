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
model.fit(train_x, train_y, n_epoch=10000, batch_size=8, show_metric=True)
model.save('model.tflearn')

def get_tf_record(sentence):
   global words
   w = token.tokenize(sentence)
   sentence_words = [stemmer.stem(word.lower()) for word in w]
   bow = [0]*len(words)
   for s in sentence_words:
      for i,w in enumerate(words):
         if w == s:
            bow[i] = 1
   return(np.array(bow))
def loadFile2(sFilename):
	'''Given a file name, return the contents of the file as a string.'''
	f = open("inputData/"+ sFilename, "r")
	sTxt = f.read()
	f.close()
	return sTxt

print (model.predict([get_tf_record("the weather is nice")]))
test = []
truePos = 0
falsePos = 0
falseNeg = 0
iFileList = []
filecount = 0
acc = 0
for fFileObj in os.walk("inputData/"):
    iFileList = fFileObj[2]
    break
for file in iFileList:
    inFile = loadFile2(file)
    parseName = file.split("_")
    if parseName[1] == "sar":
        #first is true value, second is classified value
        trial = ("sarcastic", categories[np.argmax(model.predict([get_tf_record(inFile)]))])
        test.append(trial)
    elif parseName[1] == "nsar":
        trial = ("regular", categories[np.argmax(model.predict([get_tf_record(inFile)]))])
        test.append(trial)
    print(trial)
    filecount += 1
    print(filecount)
for item in test:
    if item[0] == "sarcastic" and item[1] == "sarcastic":
        truePos += 1
        acc += 1
    elif item[0] == "sarcastic" and item[1] == "regular":
        falseNeg += 1
    elif item[0] == "regular" and item[1] == "sarcastic":
        falsePos += 1
    elif item[0] == "regular" and item[1] == "regular":
        acc += 1
accu = float( acc / len(iFileList))
precision = float(truePos)/(truePos + falsePos)
recall = float(truePos)/(truePos + falseNeg)
fmeasure = (2 * precision * recall) / (precision + recall)
print("Precision: ")
print(precision)
print("Recall: ")
print(recall)
print("f-measure: ")
print(fmeasure)
print("Accuracy: ")
print(accu)

