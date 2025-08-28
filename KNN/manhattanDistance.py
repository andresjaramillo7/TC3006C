def manhattanDistance(list1, list2):
    """
        Compute the Manhattan distance between two lists
        
        Parameters:
            list1 (list): The first list of numbers.
            list2 (list): The second list of numbers.
    """
    return sum(abs(a - b) for a, b in zip(list1, list2))