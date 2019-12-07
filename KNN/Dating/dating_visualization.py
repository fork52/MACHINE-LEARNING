import numpy as np
import pprint as pr
import matplotlib as mpl
import matplotlib.pyplot as plt

def file2matrix(filename):
	'''Function that takes a filename string as input and returns a nx3 matrix and a n-label vector '''

	fr = open(filename)
	numberOfLines = len(fr.readlines())			#find the no. of lines
	returnMat = np.zeros((numberOfLines,3))     
	classLabelVector = []

	fr = open(filename)													
	index = 0						#start from line 0 in the list
	for line in fr.readlines():
		line = line.strip() 			#Remove \n from every line
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]				#first three elements are 
		classLabelVector.append(int(listFromLine[-1]))      #last element in a line is label
		index += 1
	return returnMat,classLabelVector

dataSet,labelVector=file2matrix('datingTestSet2.txt')

# pr.pprint(dataSet[0:20])
pr.pprint(labelVector[0:20])

fig = mpl.pyplot.figure()
ax = fig.add_subplot(111) #There is only one subplot in the window


colorList = []

for i in labelVector:
	if i==1:
		colorList.append('tab:blue')
	elif i==2:
		colorList.append('tab:green')
	elif i==3:
		colorList.append('tab:red')

ax.scatter(x=dataSet[:,1], y=dataSet[:,2],s= 15*np.array(labelVector),c=colorList )
#ax.scatter(x_list,y_list,s=Thickness_list,color_list)

# U CAN TRY DIFFERENT x vs y plot for ur data
ax.legend(['color1','color2','color3'])

mpl.pyplot.title('DATING SITE')
mpl.pyplot.xlabel('% OF Time spent playing videogames')
mpl.pyplot.ylabel('Litres of ice cream consumed per week')
mpl.pyplot.grid(True)

# pr.pprint( 15.0*np.array(labelVector))
mpl.pyplot.show()