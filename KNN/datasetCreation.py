import numpy as np
import codecs
import os

base_path = os.path.dirname(os.path.abspath(__file__))
print(base_path)

feat1 = list(np.random.normal(0, 0.2, 100000))
feat2 = list(np.random.normal(0, 0.2, 100000))
label = list(np.random.choice(["A", "B"], size = 100000, p = [0.5, 0.5]))

print("save training data to file")
with codecs.open(base_path + "train.txt", "w", "UTF-8") as f:
    for f1, f2, l in zip(feat1, feat2, label):
        f.write(str(f1) + "," + str(f2) + "," + str(l) + "\n")
        
print("Create test samples")

feat1_test = list(np.random.normal(0, 0.2, 100000))
feat2_test = list(np.random.normal(0, 0.2, 100000))
label_test = list(np.random.choice(["A", "B"], size = 100000, p = [0.5, 0.5]))

print ("save test data to file")
with codecs.open(base_path + "test.txt", "w", "UTF-8") as f:
    for f1, f2, l in zip(feat1_test, feat2_test, label_test):
        f.write(str(f1) + "," + str(f2) + "," + str(l) + "\n")