import codecs
import os
from classification import classify
import matplotlib.pyplot as plt

training = []
test = []
trainingLabels = []
testLabels = []
k = 8

base_path = os.path.dirname(os.path.abspath(__file__))

print ("Loading training data from file")
with codecs.open(os.path.join(base_path, "training.txt"), "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        feat = [float(x) for x in elements[:-1]]
        label = elements[-1]
        training.append(feat)
        trainingLabels.append(label)

print ("Loading test data from file")
with codecs.open(os.path.join(base_path, "test.txt"), "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        feat = [float(x) for x in elements[:-1]]
        label = elements[-1]
        test.append(feat)
        testLabels.append(label)

print("Apply the KNN approach over test samples")
correctPredictions = 0
TotalPredictions = 0
for x, y in zip(test, testLabels):
    TotalPredictions += 1
    predicted = classify(x, training, trainingLabels, k)
    if predicted == y:
        correctPredictions += 1
    print("Predicted: " + str(predicted) + ", realValue: " + str(y))

print("Model accuracy: " + str(correctPredictions / TotalPredictions * 100) + "%")