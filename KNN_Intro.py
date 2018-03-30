import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd


# Load Data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Replace the question marks with a number
df.replace('?', -99999, inplace=True)

# Drop the ID columns
df.drop(['id'], 1, inplace=True)

# Create our features
X = np.array(df.drop(['class'], 1))

# Create our labels
y = np.array(df['class'], 1)

# Split the data into training and testing parts
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Define the classifier
classifier = neighbors.KNeighborsClassifier

# Train the data
classifier.fit(X_train, y_train)

# Test the data
accuracy = classifier.score(X_test, y_test)

print(accuracy)


# Create sample data
example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = classifier.predict(example_measures)

print()

