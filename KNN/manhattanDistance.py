def manhattanDistance(list1, list2):
    return sum(abs(a - b) for a, b in zip(list1, list2))