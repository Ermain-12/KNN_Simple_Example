import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

# Create a hard-coded data-set with features which correspond to the class of k
dataset = {
                'k': [[1, 2], [2, 3], [3, 1]],
                'r': [[6, 5], [7, 7], [8, 6]]
          }

new_features = [5, 7]

"""
for i in dataset:
    # Cycle through the pairs of data
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
"""

'''
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1])
plt.show()
'''


def k_nearest_neighbors(data, predict, k=3):
    if len(data) <= k:
        warnings.warn("k is set to a value less than the total voting groups")

    distances = []

    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            # Add each distance to the a group and specify the group in the following column
            distances.append([euclidean_distance, group])

    # Sort the distances and include on the distances up to k
    votes = [i[1] for i in sorted(distances)[: k]]
    print(Counter(votes).most_common(1))
    votes_result = Counter(votes).most_common(1)[0][0]

    return votes_result


# Test our KNN algorithm
result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)