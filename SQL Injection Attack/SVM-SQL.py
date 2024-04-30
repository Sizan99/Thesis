import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv('datasetSQL.csv')

# Preprocess data
X = df['Query'].values
y = df['Label'].values

# Vectorize text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

length_train = X_train.shape[0]
length_test = X_test.shape[0]
print("Number of instances in the training set:", length_train)
print("Number of instances in the testing set:", length_test)

# Construct SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Train model
svm_model.fit(X_train, y_train)

# Make predictions
predictions = svm_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, TN, FP, FN
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Print metrics
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)
print("Recall:", recall)
print("Precision:", precision)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

