import trees_classifier
import pickle
import pprint as pr


def storeTree(inputTree,filename):
	file = open(filename,'wb')
	pickle.dump( inputTree , file)
	file.close()

def grabTree(filename):
	fr = open(filename,'rb')
	return pickle.load(fr)

if __name__ == '__main__':
	# dataSet,labels,colNames=trees_classifier.createDataSet()
	# root = trees_classifier.createTree(dataSet,labels,colNames)
	# pr.pprint(root,indent=3,width=80)
	# storeTree(root,'GoingOut.txt')
	fetchedTree = grabTree('GoingOut.txt')
	pr.pprint(fetchedTree)