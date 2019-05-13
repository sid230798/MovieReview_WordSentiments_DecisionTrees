'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 18/8
Purpose :- 1. Structure for Node of Decision Tree	

'''


#Structure For Node of Tree

class TreeNode : 

	def __init__(self):
		self.attribute = -1
		self.word = "No Word Found"
		self.entropy = -1
		self.label = 0
		self.left_neg = None
		self.right_pos = None
		self.isLeaf = False
		self.examples_pos = 0
		self.examples_neg = 0

	def __str__(self):

		Value = "Attribute => " + str(self.attribute) + " Word =>" + self.word + " Entropy=>" +str(self.entropy)+" ISLEAF=>" + str(self.isLeaf) + " Label=>" + str(self.label)+ " GoodExam=>" + str(self.examples_pos) + " NegExamples=>" + str(self.examples_neg)


		return Value
