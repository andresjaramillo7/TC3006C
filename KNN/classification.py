from operator import itemgetter
import random

def classify(testList, trainingLists, trainingLabels, k, distance_fn):
    distances = []
    for trainingList, label in zip (trainingLists, trainingLabels):
        value = distance_fn(testList, trainingList)
        distances.append((value, label))
    distances.sort(key=itemgetter(0))
    votelabels = []
    for x in distances[:k]:
        votelabels.append(x[1])
    counts = {}
    for label in votelabels:
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return random.choice(candidates)