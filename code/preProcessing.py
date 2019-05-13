'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 18/8
Purpose :- 1. Select 1000 Random Reviews
	   2. Select 1000 words Randomly of high Sentiments.
	   3. Create new Train File	

'''

import operator
import random
import re
from reviewFeaturedVector import ReviewObject
from algoTrainID3 import run

#-------------------------------------------------------------------------------------------------------

#Create Object of Feature Vector

def createFeatureVector(review,wordDict) :
	
	listIndices = re.findall('\d+',review)
	reviewChoice = int(listIndices[0])

	if(reviewChoice >= 7):
		reviewChoice = 1
	else:
		reviewChoice = -1
	
	Object = ReviewObject(reviewChoice,len(wordDict))
	count = 1
	while(count < len(listIndices)):
		
		indexValue = int(listIndices[count])

		if(indexValue in wordDict) :
			Object.attribute[wordDict[indexValue]] = int(listIndices[count+1])

		count += 2
	
	return Object

#-------------------------------------------------------------------------------------------------------

#Feature Extaraction of Top and Bottom 2500 words

fileWordSentiments = open('DataSet/imdbEr.txt','r') 
fileWordIndex = open('DataSet/imdb.vocab','r')

indexSentiments_pos = dict()
indexSentiments_neg = dict()  
wordIndex = dict()

#Reading File and Savinf Sentiments in Dictionary
for index,line in enumerate(fileWordSentiments):

	sentiment = float(line)
	if(sentiment >= 0):
		indexSentiments_pos[index] = sentiment
	else:
		indexSentiments_neg[index] = sentiment

for index,line in enumerate(fileWordIndex):

	wordIndex[index] = line.strip()


#Sort and get Top 2500 and Bottom 2500 words
n_words = int(input('Please Enter number of Attributes to Extract from file :- '))

n_words = int(n_words/2)

indexSentiments_pos = sorted(indexSentiments_pos.items(),key = operator.itemgetter(1),reverse = True)[:n_words]
indexSentiments_neg = sorted(indexSentiments_neg.items(),key = operator.itemgetter(1))[:n_words]

#Printing Word and SentimentList in file and Creating Attribute Vector

fileOutputAttribute = open('selected-feature-indices.txt','w')
featureAttribute = dict()

indexAttr = 0
for tuples in indexSentiments_pos:
	featureAttribute[tuples[0]] = indexAttr
	indexAttr += 1
	fileOutputAttribute.write(str(tuples[0]) + " " + str(tuples[1]) + " " +wordIndex[tuples[0]] + '\n')

for tuples in indexSentiments_neg:
	featureAttribute[tuples[0]] = indexAttr
	indexAttr += 1
	fileOutputAttribute.write(str(tuples[0]) + " " + str(tuples[1]) + " " +wordIndex[tuples[0]] + '\n')

#--------------------------------------------------------------------------------------------------------------

# Selecting Random Reviews from Training Set and Test Set

fileReviewTrain = open('DataSet/labeledBowTrain.feat','r')
fileReviewTest = open('DataSet/labeledBowTest.feat','r')

#List for Stroing Positive and Negative Reviews
trainReviewGood = list()
trainReviewBad = list() 
testReviewGood = list()
testReviewBad = list() 

#File Reading from Training Set
for line in fileReviewTrain : 

	reviewChoice = int(re.findall('\d+',line)[0])

	if(reviewChoice >= 7):
		trainReviewGood.append(line)
	else:
		trainReviewBad.append(line)

#FIle Read from test Set
for line in fileReviewTest : 

	reviewChoice = int(re.findall('\d+',line)[0])

	if(reviewChoice >= 7):
		testReviewGood.append(line.strip())
	else:
		testReviewBad.append(line.strip())	

#Shuffling List for Random Selection
random.shuffle(trainReviewGood)
random.shuffle(trainReviewBad)
random.shuffle(testReviewGood)
random.shuffle(testReviewBad)

#Selecting Top 500 from each set
trainReviewGood = trainReviewGood[:500]
trainReviewBad = trainReviewBad[:500]
testReviewGood = testReviewGood[:500]
testReviewBad = testReviewBad[:500]

#------------------------------------------------------------------------------------------------------------------

trainFeatureVector = list()
testFeatureVector  = list()
wordFeature = dict()


#Create Feature Vector for Training Set and Test Set
for vector in trainReviewGood:
	trainFeatureVector.append(createFeatureVector(vector,featureAttribute))

for vector in trainReviewBad:
	trainFeatureVector.append(createFeatureVector(vector,featureAttribute))

for vector in testReviewGood:
	testFeatureVector.append(createFeatureVector(vector,featureAttribute))

for vector in testReviewBad:
	testFeatureVector.append(createFeatureVector(vector,featureAttribute))


for key,val in featureAttribute.items():
	wordFeature[val] = wordIndex[key]

#Writing Feature Vector For Test Set and Train Set
fileOutputTrainSet = open('TrainSet.txt','w')

for Object in trainFeatureVector:
	fileOutputTrainSet.write(str(Object))

fileOutputTestSet = open('TestSet.txt','w')

for Object in testFeatureVector:
	fileOutputTestSet.write(str(Object))

Attributes = dict()

for j in range(0,len(featureAttribute)):
	Attributes[j] = j

#-------------------------------------------------------------------------------------------------------------------

#run(trainFeatureVector,Attributes,wordFeature,testFeatureVector)
