'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 18/8
Purpose :- Represents Feature Vector Object	

'''

#Defining a Review Object Which contains a list of indices and it's Review
class ReviewObject :

	def __init__(self,review,size):

		self.review = review
		self.attribute = dict()
		for i in range(0,size):
			self.attribute[i] = 0

	

	def __str__(self):

		string = str(self.review)
		for i in range(0,len(self.attribute)):
			string += " "+str(self.attribute[i])		

		string += '\n'
		return string
