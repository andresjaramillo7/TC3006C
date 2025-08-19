import codecs
import os
from classification import classify

training = []
test = []
trainingLabels = []
testLabels = []
k = 100

base_path = os.path.dirname(os.path.abspath(__file__))

print ("Loading training data from file")
with codecs.open(base_path + "train.txt", "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        training.append([float(elements[0]), float(elements[1])])
        trainingLabels.append(elements[2])
        
print ("Loading test data from file")
with codecs.open(base_path + "test.txt", "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        test.append([float(elements[0]), float(elements[1])])
        testLabels.append(elements[2])
        
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