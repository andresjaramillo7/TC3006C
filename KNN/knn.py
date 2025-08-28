import codecs
import os
from classification import classify
from euclideanDistance import euclideanDistance
from manhattanDistance import manhattanDistance
from cosineDistance import cosineDistance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Lists to hold the training and test data
training = [] # Training features
test = [] # Test features
trainingLabels = [] # Training labels
testLabels = [] # Test labels
k_values = list(range(3, 42)) # K values to test, from 3 to 41
# Dictionary of distance functions
distance_fns = {
    "manhattan": manhattanDistance,
    "euclidean": euclideanDistance,
    "cosine": cosineDistance,
}

base_path = os.path.dirname(os.path.abspath(__file__)) # Base path for data files

print ("Loading training data from file")
with codecs.open(os.path.join(base_path, "training.txt"), "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        feat = [float(x) for x in elements[:-1]] # Features are all but last element
        label = elements[-1] # Label is the last element
        training.append(feat)
        trainingLabels.append(label)

print ("Loading test data from file")
with codecs.open(os.path.join(base_path, "test.txt"), "r", "UTF-8") as f:
    for line in f:
        elements = (line.rstrip('\n')).split(",")
        feat = [float(x) for x in elements[:-1]] # Features are all but last element
        label = elements[-1] # Label is the last element
        test.append(feat)
        testLabels.append(label)

print("Apply the KNN approach over test samples")
results_by_metric = {} # Dictionary to hold results by distance metric

"""
    This segment iterates over all distance functions (euclidean, manhattan and cosine) and applies
    the KNN algorithm for each combination of k and distance function.
"""
for name, dist_fn in distance_fns.items():
    print(f"\n --- Distance: {name} ---")
    accuracy = {k: {"correct": 0, "total": 0} for k in k_values} # Dictionary to hold accuracy counts
    # Classify each test instance for each k
    for x, y in zip(test, testLabels):
        for k in k_values:
            predicted = classify(x, training, trainingLabels, k, dist_fn) # Classify the test instance
            if predicted == y:
                accuracy[k]["correct"] += 1
            accuracy[k]["total"] += 1
            print(f"k={k}, Predicted: {predicted}, Real: {y}")

    # Calculate accuracies and finds the best k from the dictionary results
    accuracies = {k: accuracy[k]["correct"] / accuracy[k]["total"] * 100 for k in k_values} # Calculate accuracy for each k
    best_k = max(accuracies, key=accuracies.get) # Find the best k
    best_acc = accuracies[best_k] # Find the best accuracy

    print(f"\nBest k for {name} distance = {best_k}, Accuracy = {round(best_acc, 2)}%")
    # Store the results in the dictionary
    results_by_metric[name] = {"best_k": best_k, "best_acc": best_acc, "accuracies": accuracies}

"""
    This segment of the code selects overall the best metric based on the highest accuracy and its corresponding k.
"""
best_metric = max(results_by_metric, key=lambda m: results_by_metric[m]["best_acc"]) # Best metric based on accuracy
best_k = results_by_metric[best_metric]["best_k"] # Best k for the best metric
best_acc = results_by_metric[best_metric]["best_acc"] # Best accuracy for the best metric
accuracies = results_by_metric[best_metric]["accuracies"] # Accuracies for the best metric
best_dist_fn = distance_fns[best_metric] # Best distance function for the best metric

print(f"\nBest Distance Function: {best_metric}, Best k = {best_k}, Best Accuracy = {round(best_acc, 2)}%")

"""
    This segment of the code plots accuracy vs k for the best distance function.
"""
acc_list = [accuracies[k] for k in k_values] # Get accuracy list for the best metric

plt.figure(figsize=(6, 5))
plt.plot(k_values, acc_list, marker='o')
plt.title(f"K vs. Accuracy ({best_metric})")
plt.xlabel("k")
plt.ylabel("Accuracy (%)")


"""
    This segment creates the confusion matrix for the best distance function and the best k.
"""
pos_label = "Present" # Positive label
neg_label = "Absence" # Negative label
label_to_idx = {neg_label: 0, pos_label: 1} # Map labels to indices
cm = [[0, 0], [0, 0]] # Confusion matrix
# Populate the confusion matrix
for x, y_true in zip(test, testLabels):
    y_pred = classify(x, training, trainingLabels, best_k, best_dist_fn)
    i = label_to_idx[y_true] # True label index
    j = label_to_idx[y_pred] # Predicted label index
    cm[i][j] += 1
tn, fp = cm[0][0], cm[0][1] # True negatives and false positives
fn, tp = cm[1][0], cm[1][1] # False negatives and true positives

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest') # Display confusion matrix
plt.title(f"Confusion Matrix (k={best_k}, Distance={best_metric})")
plt.xticks([0,1], [f"Predicted {neg_label}", f"Predicted {pos_label}"]) # Predicted labels
plt.yticks([0,1], [f"True {neg_label}", f"True {pos_label}"]) # True labels

# Add the counts to the confusion matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha="center", va="center")
plt.tight_layout() # Adjust layout so that all labels are visible

"""
    This segment creates a heatmap to find the correlation between features using the training dataset,
    to understand how features relate to each other.
"""
# Feature names from the dataset (13 in total)
feat_names = [
    "age", "sex", "chest pain", "blood pressure", "serum cholesterol", "fasting glucose",
    "resting ECG", "heart rate", "exercise-induced angina", "exercise-induced ST-segment depression relative to rest",
    "ST segment of maximum exercise", "number of major vessels colored by fluoroscopy", "thalassemia presence"
]

# Build Dataframe for feature correlation
df = pd.DataFrame(training, columns=feat_names)
df["label"] = trainingLabels # Add labels to the Dataframe

plt.figure(figsize=(6,5))
# Create a heatmap to visualize feature correlations excluding the label
sns.heatmap(df.iloc[:, :-1].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")

plt.show() # Display the plots