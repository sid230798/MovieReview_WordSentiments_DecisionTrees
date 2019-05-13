'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 18/8
Purpose :- 1. Train the selected reviews for Decision Tree.	

'''

from reviewFeaturedVector import ReviewObject
from TreeNode import TreeNode
import math
import copy
import sys
import random
import operator

#---------------------------------------------------------------------------------

#Calculate Entropy of Examples

def getEntropy(countGood,tot):

	prop = countGood*1.0/tot

	if(countGood == 0 or countGood == tot):
		return 0

	prop = -(prop*math.log(prop,2) + (1-prop)*math.log((1-prop),2))

	return prop

#---------------------------------------------------------------------------------

#Calculate minEntropy and index By Weighted Entropy
def getMinWeightedEntropy(Examples,Attribute):

	#Assigning min Entropy and index
	minEntropy = -1.0
	index = -1

	#Iterating through Attribute for each of it value
	for j in Attribute:

		#Initialise No of Examples of Positive and Negative
		countGood_0 = 0
		countBad_0 = 0
		countGood_1 = 0
		countBad_1 = 0

		#List through All examples for each attribute
		for Object in Examples : 

			if(Object.review == 1):
				if(Object.attribute[j] == 0):
					countGood_0 += 1
				else:
					countGood_1 += 1
			else:
				if(Object.attribute[j] == 0):
					countBad_0 += 1
				else:
					countBad_1 += 1

		totExample_0 = countGood_0 + countBad_0
		totExample_1 = countGood_1 + countBad_1

		if(totExample_0 == 0 or totExample_1 == 0):
			continue

		#Weighted Entropy of for each attribute
		entropy_j = (totExample_0*getEntropy(countGood_0,totExample_0) + totExample_1*getEntropy(countGood_1,totExample_1))/(1.0*(totExample_0 + totExample_1))

		if(entropy_j <= minEntropy or minEntropy == -1):
			minEntropy = entropy_j
			index = j

	return index,minEntropy

#---------------------------------------------------------------------------------------------

#Returns Set of Examples for particular attribute
def getListofExamples(Examples,index):

	Examples_0 = list()
	Examples_1 = list()

	for Object in Examples:
		if(Object.attribute[index] == 0):
			Examples_0.append(Object)

		else:
			Examples_1.append(Object)

	return Examples_0,Examples_1

#------------------------------------------------------------------------------------------------

#Function to count +reviews
def makeCountOfGoodReviews(Examples):

	countGood = 0
	for Object in Examples:
		if(Object.review == 1):
			countGood += 1

	return countGood

#-------------------------------------------------------------------------------------------------

#Function for Pruining the Tree for Accuracy improvement

def prune(Root):


	#Error is number of misclassified
	if(Root.isLeaf == True):
		if(Root.label == 1):
			return Root.examples_neg
		else:
			return Root.examples_pos
	else:
		#Total error is Root.left error plus RRoot.right error
		error = prune(Root.left_neg) + prune(Root.right_pos)

		#If error of parent is less than root child don't prune
		#Else prune
		if(error < min(Root.examples_pos,Root.examples_neg)):
			return error
		else:
			Root.isLeaf = True
			#Assign Label as maxLabels
			if(Root.examples_pos >= Root.examples_neg):
				Root.label = 1
				return Root.examples_neg
			else:
				Root.label = -1
				return Root.examples_pos

#-------------------------------------------------------------------------------------------------


#This Recursive implementation of ID3 Algorithm
#It takes set of Examples,Attributes and WordDictionar

def funcID3(Examples,Attributes,wordIndex):

	Root = TreeNode()
	
	#Count no. of GoodExamples and Bad Examples
	countGood = makeCountOfGoodReviews(Examples)
	countBad = len(Examples) - countGood
	Root.examples_pos = countGood
	Root.examples_neg = len(Examples) - countGood

	maxLabel = 1 if countGood >= countBad else -1

	#If Entropy at this node is 0.1 Return node with maxLabel 
	

	if(getEntropy(countGood,len(Examples)) <= 0.0):
		Root.isLeaf = True
		Root.label = maxLabel
		return Root

	else:
		#Calls For function calculate minGain
		bestAttr,entropy = getMinWeightedEntropy(Examples,Attributes)

		#It's -1 if InfoGain = 0 for all attributes So it Gives maxLabel
		if(bestAttr == -1 or entropy <= 0.0):
			Root.isLeaf = True
			Root.label = maxLabel
			return Root
		else:
			#If all conditions turns correct it splits node by its value and recursively call 
			#childs of Root
			Root.attribute = bestAttr
			Root.entropy = entropy
			Root.word = wordIndex[bestAttr]
			Examples_0,Examples_1 = getListofExamples(Examples,bestAttr)
			
			#DeepCopy for Just copying Values
			Attributes_0 = copy.deepcopy(Attributes)

			del Attributes_0[bestAttr]

			Root.left_neg = funcID3(Examples_0,Attributes_0,wordIndex)
			Root.right_pos = funcID3(Examples_1,Attributes_0,wordIndex)


			return Root

#--------------------------------------------------------------------------------------------------------

#Its the Predictor Function Accept TestData and return found Label
def predict(TestData,Root):


	#If Root is Leaf then return its Label as we found prediction
	if(Root.isLeaf == True):
		return Root.label

	k = Root.attribute

	#Check for which Branch to select for attribute value
	if(TestData.attribute[k] == 0):
		return predict(TestData,Root.left_neg)
	else:
		return predict(TestData,Root.right_pos)

#--------------------------------------------------------------------------------------------------------

def getDictionaryofWords(Root,words):

	#If Root is Leaf then return its Label as we found prediction
	if(Root == None):
		return

	if(Root.isLeaf == True):
		return

	if Root.word in words.keys():
		words[Root.word] += 1
	else:
		words[Root.word] = 1

	getDictionaryofWords(Root.left_neg,words)
	getDictionaryofWords(Root.right_pos,words)

#--------------------------------------------------------------------------------------------------------

#Function returns Random Sample of Attribute and corresponding Words

def getRandomSubset(Attribute,wordIndex,N_Samples):

	#For Equal number of Positive Sentiment Attribute and Negative Sentiment Attribute
	Random_pos = list()
	Random_neg = list()

	half_attr = int(len(Attribute)/2)
	for i in range(0,half_attr):
		Random_pos.append(i)

	for i in range(half_attr,len(Attribute)):
		Random_neg.append(i)

	#Shuffle Random Index and Select top Samples/2
	random.shuffle(Random_pos)
	random.shuffle(Random_neg)

	half_sample = int(N_Samples/2)
	Random_pos = Random_pos[:half_sample]
	Random_neg = Random_neg[:half_sample]

	#Merge two list by summation
	Random = Random_pos + Random_neg

	#Create a Dictionary as Standard Input for ID3 algo
	Attribute_dict = dict()
	wordIndexDict = dict()

	for index in Random:
		Attribute_dict[index] = index
		wordIndexDict[index] = wordIndex[index]

	#print(Attribute_dict)
	#print(wordIndexDict)
	return Attribute_dict,wordIndexDict

#--------------------------------------------------------------------------------------------------------

#Function For Feature Bagging to create Random Forest

def FeatureBagging(Examples,Attribute,wordIndex,ExamplesTest,N_Trees,N_Attr):

	#RootList to Store Nodes of each tree
	RootList = list()

	N_Attr = min(len(Attribute),N_Attr)
	ListAns = list()
	for i in range(0,N_Trees):

		#Getting Random 1000 Attributtes ans coresponding words
		Attributes_Sample, wordIndexSample = getRandomSubset(Attribute,wordIndex,N_Attr)

		print("Sample Started to Train")

		#Creates Root for each Attribute set
		Root = funcID3(Examples,Attributes_Sample,wordIndexSample)

		#Calculating Height and Leafs for Testing Purpose
		MaxHeight,Leafs = height(Root)

		print("Height of Decision Tree : " + str(MaxHeight))
		print("No of Leafs : " + str(Leafs))

		print("---------------------------------------------------")
		RootList.append(Root)


	
		#Code to Test accuracy of Random Forest on Test Examples
		countTrue = 0
		for Object in Examples:

			#Count Number of Positive labels and Negative labels returend from
			# Tree of Random Forest
			posLabels = 0
			negLabels = 0

			#Iterating Through each Node
			for Root in RootList:
				k = predict(Object,Root)

				if(k == 1):
					posLabels += 1
				else:
					negLabels += 1

				if(k == 0):
					print("Not Possible")
	
			#Classifying by Max number of Labels
			maxLabel = 1 if posLabels >= negLabels else -1

			if(maxLabel == Object.review):
				countTrue += 1

		TrainAcc = countTrue*1.0/10
		#print("No of Examples Correctly classified : "+str(countTrue*1.0/10))

		#Code to Test accuracy of Random Forest on Test Examples
		countTrue = 0
		for Object in ExamplesTest:

			#Count Number of Positive labels and Negative labels returend from
			# Tree of Random Forest
			posLabels = 0
			negLabels = 0

			#Iterating Through each Node
			for Root in RootList:
				k = predict(Object,Root)

				if(k == 1):
					posLabels += 1
				else:
					negLabels += 1

				if(k == 0):
					print("Not Possible")
	
			#Classifying by Max number of Labels
			maxLabel = 1 if posLabels >= negLabels else -1

			if(maxLabel == Object.review):
				countTrue += 1

		TestAcc = countTrue*1.0/10
		
		ans = "No. of Trees = " +str(i+1) + " Train-acc = "+str(TrainAcc)+" Test-Acc"+str(TestAcc)
		ListAns.append(ans)


	for string in ListAns:
		print(string) 
		#print("No of Examples Correctly classified : "+str(countTrue*1.0/10))
	

#-------------------------------------------------------------------------------------------------

#Prints the PreOrder Traversal of Tree Just for Verification
def preOrder(Root) : 

	if(Root.isLeaf == True):
		print(Root)
		return

	print(Root)

	if(Root.left_neg == None):
		print("Got this as None")
	else:
		preOrder(Root.left_neg)

	if(Root.right_pos == None):
		print("Got this as None")
	else:
		preOrder(Root.right_pos)


#--------------------------------------------------------------------------------------------------------

#Calculates Depth of Decision Tree and Count Leafs
def height(Root):

	if(Root.isLeaf == True):
		return 0,1

	#Accept tuples in returning from height
	hMax1,count1 = height(Root.left_neg)
	hMax2,count2 = height(Root.right_pos)

	hMax = 1+max(hMax1,hMax2)
	count = count1+count2

	return hMax,count

#--------------------------------------------------------------------------------------------------------

#Function add Random noise to Training Set

def addNoise(Examples,noise_percent):

	#Calculate examples to modify
	n_examples = int(noise_percent * len(Examples)/100)

	Examples_Change = copy.deepcopy(Examples)

	for i in range(0,n_examples):

		#Choose Random Integer for Review to invert
		random_int = random.randint(0,(len(Examples)-1))

		#print(random_int)
		Examples_Change[random_int].review = -1*Examples_Change[random_int].review

	
	return Examples_Change

#--------------------------------------------------------------------------------------------------------

def shortHeight(Root,height):

	if(Root == None):
		return Root

	if(Root.isLeaf == True):
		return Root

	if(height == 30):
		maxLabel = 1 if Root.examples_pos >= Root.examples_neg else -1
		Root.isLeaf = True
		Root.label = maxLabel
		Root.left_neg = Root.right_pos = None
		return Root

	Root.left_neg = shortHeight(Root.left_neg,height+1)
	Root.right_pos = shortHeight(Root.right_pos,height+1)

	return Root

#--------------------------------------------------------------------------------------------------------

#Alternatively we could Read files created by preprocessing and Form this Dictionary
def run(Examples,Attribute,wordIndex,ExamplesTest,choice):
	

	print('Training of Data Started -----------------')

	Root = funcID3(Examples,Attribute,wordIndex)

	MaxHeight,Leafs = height(Root)
	print('Decision Tree Created --------------------')

	print("Height of New Decision Tree : " + str(MaxHeight))
	countTrue = 0
	for i in Examples:
		
		k = predict(i,Root)
		#print('Got this as Review : ' + str(k))
		if(k == i.review):
			countTrue += 1

	print("Percent Correctly Splitted on TrainSet: "+str(countTrue*1.0/10))

	countTrue = 0
	for i in ExamplesTest:
		
		k = predict(i,Root)
		#print('Got this as Review : ' + str(k))
		if(k == i.review):
			countTrue += 1

	print("Percent Correctly Splitted on TestSet: "+str(countTrue*1.0/10))

	#----------------------------------------------------------------------
	if(choice == 1):
		words = dict()
		getDictionaryofWords(Root,words)
		sortedDict = sorted(words.items(),key = operator.itemgetter(1),reverse = True)

		print("Number of times Attribute used in Splitting ........")
		for tuples in sortedDict:
			print(tuples[0] + " -> " + str(tuples[1]))

	elif(choice == 2):
		error = float(input("Enter the percentage of labels to invert :- "))
		#print(error)
		
		print("Creating Error Samples ------")
		ExamplesNoise = addNoise(Examples,error)
		print("Error Samples Created , Training on error Samples Started ------")
		Root1 = funcID3(ExamplesNoise,Attribute,wordIndex)

		print("Training Completed")

		MaxHeight,Leafs = height(Root1)
		countTrue = 0
		for i in ExamplesTest:
		
			k = predict(i,Root1)
			#print('Got this as Review : ' + str(k))
			if(k == i.review):
				countTrue += 1

		print("Height of New Decision Tree : " + str(MaxHeight))
		print("Percent Correctly Splitted For Test: "+str(countTrue*1.0/10))
		
	elif(choice == 3):
		
		prune(Root)
		print("After Pruning ---------------")
		MaxHeight,Leafs = height(Root)

		print("Height of Decision Tree : " + str(MaxHeight))
	
		countTrue = 0
		for i in Examples:
		
			k = predict(i,Root)
			#print('Got this as Review : ' + str(k))
			if(k == i.review):
				countTrue += 1

		print("Percent Correctly Splitted TrainSet: "+str(countTrue*1.0/10))

		countTrue = 0
		for i in ExamplesTest:
		
			k = predict(i,Root)
			#print('Got this as Review : ' + str(k))
			if(k == i.review):
				countTrue += 1

		print("Percent Correctly Splitted TestSet: "+str(countTrue*1.0/10))
	
	elif(choice == 4):
		
		nTrees = int(input("Enter Max Number of Trees to accomodate in Forest :- "))
		nAttr = int(input("Enter number of attributes in Each forest :- "))
	
		print("Started Random Forest ----------------------> ")
		FeatureBagging(Examples,Attribute,wordIndex,ExamplesTest,nTrees,nAttr)
		print("Ended Random Forest ------------------------>")
		
	else:
		print("No Function exist")
	
#----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	#Reading Input Set from Train and Test file
	fileOpenTrainSet = open('TrainSet.txt','r')
	fileOpenTestSet = open('TestSet.txt','r')

	#Create Train Set from file 
	ExamplesTrain = list()
	for line in fileOpenTrainSet:

		line = line.strip()
		array_val = line.split()

		dict_size = len(array_val) -1

		Object = ReviewObject(int(array_val[0]),dict_size)
		for i in range(1,len(array_val)):
			Object.attribute[i-1] = int(array_val[i])

		ExamplesTrain.append(Object)

	#Create Test Set from File
	ExamplesTest = list()
	for line in fileOpenTestSet:

		line = line.strip()
		array_val = line.split()

		dict_size = len(array_val) -1
		Object = ReviewObject(int(array_val[0]),dict_size)
		for i in range(1,len(array_val)):
			Object.attribute[i-1] = int(array_val[i])

		ExamplesTest.append(Object)

	#Create Attribute and wordIndex Dict by selected feature file
	fileWords = open('selected-feature-indices.txt','r')
	Attributes = dict()
	wordIndex = dict()

	for index,line in enumerate(fileWords):

		line = line.strip()
		array_val = line.split()

		Attributes[index] = index
		wordIndex[index] = array_val[2]

	#print(len(ExamplesTrain))
	run(ExamplesTrain,Attributes,wordIndex,ExamplesTest,int(sys.argv[1]))
	#getRandomSubset(Attributes,wordIndex,100)
	#addNoise(ExamplesTrain,0.5)
#---------------------------------------------------------------------------------------------------------------
