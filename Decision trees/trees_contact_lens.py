import trees_classifier,tree_plotter
import pprint as pr

dataFile = open('lenses.txt')

l = dataFile.readlines()

dataSet1 = []
labels1=[]

for line in l:
    line = line.strip()
    line = line.split('\t')
    dataSet1.append(line[0:-1])
    labels1.append(line[-1])
colNames1 = ['age','preScript','astigmatic?','Tear-Rate']


root=trees_classifier.createTree(dataSet1,labels1,colNames1)

pr.pprint(root)

tree_plotter.createPlot(root)