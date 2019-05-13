----------------------------------------------------------------

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 27/8/18

Sys Requirement :- Python 3.x
----------------------------------------------------------------

/* To Produce processed file for Input to Algorithms */

*Change Directory to code

User@Name:~ python preProcessing.py

Input : Number of Attributes to Extract
Output : TrainSet.txt,TestSet.txt,selected-feature-indices.txt

Each file contains its feature Vectors to be used to learn ID3

*Feature Bagging is conducted on 2000 Attr so preProcessing Attributes >= 5000

----------------------------------------------------------------

/* For Training model */

User@Name:~ python algoTrainID3.py EXPT_NO

/*Each experiment is preceeded by Training model So First it trains and then carry out Experiments*/

Input :- Based on EXPT_NO following is done :

	 EXPT_NO = 1 -> Printes the frequency of Words that are used as split.
	 EXPT_NO = 2 -> Asks User How much noise to add and Retrain The Model.
	 EXPT_NO = 3 -> Pruning is conducted on Test Set and Results are Shown.
	 EXPT_NO = 4 -> Feature Bagging is Conducted asking user for Max number of Trees.

Output :- Data Returned by function.

----------------------------------------------------------------

Other Files :-

*TreeNode.py :- Contains Structure of Node in Decision tree.

*ReviewFeaturedVector.py :- Contains structure of Feature Vector
