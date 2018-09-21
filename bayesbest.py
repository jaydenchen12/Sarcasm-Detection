#created by Hongjie Chen
# 11/28/2017


import math, os, pickle, re

class Bayes_Classifier:
   def __init__(self, trainDirectory = "./"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text.'''
      self.positiveDict = {}
      self.negativeDict = {}
      if (os.path.isfile(trainDirectory + "negativeDictionary_best2.txt") and os.path.isfile(trainDirectory + "positiveDictionary_best2.txt")):
           self.positiveDict = self.load("positiveDictionary_best2.txt")
           self.negativeDict = self.load("negativeDictionary_best2.txt")
      else:
           self.train()

   def train(self):
       '''Trains the Naive Bayes Sentiment Classifier.'''
       iFileList = []

       for fFileObj in os.walk("trainData/"):
            iFileList = fFileObj[2]
            break
       for file in iFileList:
            inFile = self.loadFile(file)
            wordList = self.tokenize(inFile)
            parseName = file.split("_")
            print(parseName)
            if parseName[0] == "sar":
                for word in wordList:
                    if word not in self.positiveDict:
                        self.positiveDict[word] = 0
                    self.positiveDict[word] += 1
            elif parseName[0] == "nsar":
                for word in wordList:
                    if word not in self.negativeDict:
                        self.negativeDict[word] = 0
                    self.negativeDict[word] += 1

       self.save(self.negativeDict, "negativeDictionary_best2.txt")
       self.save(self.positiveDict, "positiveDictionary_best2.txt")


   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      texts = self.tokenize(sText);
      pPos = 0.5
      pNeg = 0.5
      probabilitiesPos = []
      probabilitiesNeg = []
      probabilitiesPos.append(math.log(pPos))
      probabilitiesNeg.append(math.log(pNeg))
      for text in texts:
          #calulating the chance that it is positive and negative
          numberPos = self.positiveDict.get(text)
          numberNeg = self.negativeDict.get(text)
          if numberPos is None and numberNeg is None:
              numberPos = 1
              numberNeg = 1
              continue
          elif numberPos is None:
              numberPos = 1
          elif numberNeg is None:
              numberNeg = 1
          numberPos = float(numberPos)
          numberNeg = float(numberNeg)
          pWordPos = numberPos / sum(self.positiveDict.itervalues())
          pWordNeg = numberNeg / sum(self.negativeDict.itervalues())
          condProbPos = numberPos / ( numberNeg + numberPos)
          condProbPosOut = (condProbPos * pWordPos) / pPos
          probabilitiesPos.append(math.log(condProbPosOut))

          condProbNeg = numberNeg / ( numberNeg + numberPos)
          condProbNegOut = (condProbNeg * pWordNeg) / pNeg
          probabilitiesNeg.append(math.log(condProbNegOut))

      totalProbNeg = sum(probabilitiesNeg)
      totalProbPos = sum(probabilitiesPos)
      if totalProbPos > totalProbNeg:
          return "sarcastic"
      elif totalProbPos < totalProbNeg:
          return "regular"
      else:
          print(totalProbNeg, totalProbPos)
          return "neutral"

   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open("trainData/"+ sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt

   def loadFile2(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open("inputData/"+ sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt


   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "wb")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()

   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "rb")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText):
      '''Given a string of text sText, returns a list of the individual tokens that
      occur in that string (in order).'''
      sText = sText.lower()
      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
               sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "" and c.strip() != "," and c.strip() != "." and c.strip() != "?" and c.strip() != "!":
               lTokens.append(str(c.strip()))

      if sToken != "":
         lTokens.append(sToken)
      modList = []
      for index in range(len(lTokens)):
         try:
             modList.append(lTokens[index] + " " + lTokens[index+1])
         except IndexError:
             modList.append(lTokens[index])
      return modList

bayes = Bayes_Classifier()
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
    inFile = bayes.loadFile2(file)
    wordList = bayes.tokenize(inFile)
    parseName = file.split("_")
    if parseName[1] == "sar":
        #first is true value, second is classified value
        trial = ("sarcastic", bayes.classify(inFile))
        test.append(trial)
    elif parseName[1] == "nsar":
        trial = ("regular", bayes.classify(inFile))
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
accu = float( acc )  / len(test)
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

