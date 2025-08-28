import math

def euclideanDistance(list1, list2):
    """
        Compute the Euclidean distance between two lists.
        
        Parameters:
            list1 (list): The first list of numbers.
            list2 (list): The second list of numbers.

    """
    sumList = 0
    for x, y in zip(list1, list2):
        sumList += (x - y) ** 2
    return math.sqrt(sumList)