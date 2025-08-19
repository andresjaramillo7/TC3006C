from operator import itemgetter
from statistics import mode
from euclideanDistance import euclideanDistance

def classify(testList, trainingLists, trainingLabels, k):
    distance = []
    for trainingList, label in zip (trainingLists, trainingLabels):
        value = euclideanDistance(testList, trainingList)
        distance.append((value, label))
    distance.sort(key=itemgetter(0))
    votelabels = []
    for x in distance[:k]:
        votelabels.append(x[1])
    return mode(votelabels)