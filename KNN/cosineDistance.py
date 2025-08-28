import math

def cosineDistance(list1, list2):
    """
        Compute the cosine distance between two lists.
        
        Parameters:
            list1 (list): The first list of numbers.
            list2 (list): The second list of numbers.
    """
    dot_product = sum(a * b for a, b in zip(list1, list2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in list1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in list2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 1.0
    return 1 - (dot_product / (magnitude1 * magnitude2))