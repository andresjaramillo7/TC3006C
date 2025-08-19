import math

def euclideanDistance(list1, list2):
    sumList = 0
    for x, y in zip(list1, list2):
        sumList += (x - y) ** 2
    return math.sqrt(sumList)