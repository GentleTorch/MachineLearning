from numpy import *
import operator
import sys
from os import listdir


def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

group,labels=createDataSet()
# print group,labels

def classfy0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5

    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndicies[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    # print classCount
    # print classCount.iteritems()

    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    # print type(sortedClassCount)
    return sortedClassCount[0][0]

print classfy0([0,0],group,labels,3)


def file2matrix(filename):
    fr=open(filename)
    arrayLines=fr.readlines()
    numberOfLines=len(arrayLines)

    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0

    for line in arrayLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]    #feature
        classLabelVector.append(listFromLine[-1])#label
        index+=1

    return returnMat,classLabelVector

datingDateMat,datingLabels=file2matrix('datingTestSet.txt')

# print datingDateMat[:5],datingLabels[:5]

def img2vector(filename):
    returnVect=zeros((1,1024))

    fr= open(filename,'r')
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)

    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)

    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)

    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])

        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classfy0(vectorUnderTest,trainingMat,hwLabels,3)
        print 'The classifier result: %d, the real answer: %d'%(classifierResult,classNumStr)

        if classNumStr!=classifierResult:
            errorCount+=1.0

    print 'Error rate: %f'%(errorCount/float(mTest))