from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import codecs

# Create lists to hold the training and test data
training = [] # Training features
test = [] # Test features
trainingLabels = [] # Training labels
testLabels = [] # Test labels
k_values = list(range(3, 42)) # K values to test, from 3 to 41
metrics = ["euclidean", "manhattan", "cosine"] # Distance metrics to test

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

results_by_metric = {} # Dictionary to hold results by distance metric

"""
    This segment of the code iterates over all distance metrics (euclidean, manhattan and cosine),
    trains a KNN classifier for each k in k_values, evaluates its accuracy, and stores the results.
"""
for metric in metrics:
    print(f"\n--- Metric: {metric} ---")
    acc_by_k = {} # Dictionary to hold accuracy by k

    for k in k_values:
        # Create and train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(training, trainingLabels)
        # Predict on the test dataset
        preds = knn.predict(test)
        # Compute accuracy (percentage of correct predictions)
        acc = accuracy_score(testLabels, preds) * 100.0
        acc_by_k[k] = acc # Store accuracy for this k

    # Find the best k for this metric
    best_k = max(acc_by_k, key=acc_by_k.get)
    best_acc = acc_by_k[best_k]
    print(f"Best k for {metric} = {best_k}, Accuracy = {round(best_acc, 2)}%")
    # Store results for this metric into the global dictionary
    results_by_metric[metric] = {"accuracies": acc_by_k, "best_k": best_k, "best_acc": best_acc}

"""
    This segment identifies the overall best distance metric based on the highest accuracy,
    and retrieves the corresponding best k and best accuracy values.
"""
best_metric = max(results_by_metric, key=lambda m: results_by_metric[m]["best_acc"])
best_k = results_by_metric[best_metric]["best_k"] # Best k for the best metric
best_acc = results_by_metric[best_metric]["best_acc"] # Best accuracy for the best metric
print(f"\nBest Distance Function: {best_metric}, Best k: {best_k}, Best Accuracy: {round(best_acc, 2)}%")

"""
    This segment creates and evaluates the KNN classifier with the best parameters to
    generate a prediction on the test dataset and build the confusion matrix.
"""
# Train the final KNN classifier with the best parameters
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn_best.fit(training, trainingLabels)
# Predict on the test dataset
preds_best = knn_best.predict(test)

# Determine the order of labels for the confusion matrix
labels_order = (
    ["Absence", "Present"] if set(testLabels) == {"Absence", "Present"} or set(trainingLabels) == {"Absence", "Present"}
    else sorted(set(trainingLabels) | set(testLabels))
)
# Build the confusion matrix
cm = confusion_matrix(testLabels, preds_best, labels=labels_order)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest') # Display the confusion matrix
plt.title(f"Confusion Matrix (k={best_k}, metric={best_metric})")

# Axis labels: Predicted labels on x-axis, True labels on y-axis
plt.xticks(range(len(labels_order)), [f"Pred {l}" for l in labels_order])
plt.yticks(range(len(labels_order)), [f"True {l}" for l in labels_order])

# Add the counts to the confusion matrix
for i in range(len(labels_order)):
    for j in range(len(labels_order)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
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

# Build DataFrame for feature correlation
df = pd.DataFrame(training, columns=feat_names)
df["label"] = trainingLabels # Add labels to the DataFrame

plt.figure(figsize=(6,5))
# Create a heatmap to visualize feature correlations excluding the label
sns.heatmap(df.iloc[:, :-1].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")

"""
    This segment generates a comparative plot to visualize the accuracy versus k for all distance metrics.
"""
plt.figure(figsize=(7, 5))
# Iterate through the results by metric and plot accuracy vs k
for metric, res in results_by_metric.items():
    ks = sorted(res["accuracies"].keys()) # Get all k values
    accs = [res["accuracies"][k] for k in ks] # Get all accuracies for the corresponding k values
    plt.plot(ks, accs, marker='o', label=f"{metric} (best k={res['best_k']})") # Label each line with the metric and best k

plt.title("K vs. Accuracy for Different Metrics")
plt.xlabel("k")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout() # Adjust layout so that all labels are visible

"""
    This segment creates a bar plot to compare the best accuracy and optimal k for each distance metric.
"""
metrics = list(results_by_metric.keys()) # Get the list of metrics (euclidean, manhattan and cosine)
best_accs = [results_by_metric[m]["best_acc"] for m in metrics] # best accuracy per metric
best_ks = [results_by_metric[m]["best_k"] for m in metrics] # best k per metric

plt.figure(figsize=(7,5))
bars = plt.bar(metrics, best_accs) # Create a bar for each metric
plt.bar_label(bars, labels=[f"{round(acc,2)}%" for acc in best_accs], padding=3) # Add accuracy labels above bars
plt.title("Best Accuracy/Performance by Metric (with Optimal k)")
plt.ylabel("Accuracy (%)")
# Expand y axis so annotations don't overlap with the top of the plot
plt.ylim(0, max(best_accs) * 1.1)
plt.tight_layout() # Adjust layout so that all labels are visible

plt.show() # Display the plots