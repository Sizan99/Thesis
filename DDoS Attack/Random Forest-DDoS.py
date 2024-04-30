import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,
                             f1_score)
from tqdm import tqdm

# Load CIC-IDS2017 dataset
df = pd.read_csv('datasetDDoS.csv')

# Remove leading and trailing whitespace from column names
df.columns = df.columns.str.strip()

# Select relevant features for input
X = df[['Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets']].values
y = df['Label'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Number of instances in the training set:", len(X_train))
print("Number of instances in the testing set:", len(X_test))

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels in y_train
y_train_encoded = label_encoder.fit_transform(y_train)

# Transform the labels in y_test
y_test_encoded = label_encoder.transform(y_test)

# Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train classifier
with tqdm(total=len(X_train_scaled), desc="Training") as pbar:
    rf_classifier.fit(X_train_scaled, y_train_encoded)
    pbar.update(len(X_train_scaled))

# Evaluate classifier
accuracy = rf_classifier.score(X_test_scaled, y_test_encoded)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = rf_classifier.predict(X_test_scaled)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate metrics
accuracy = accuracy_score(y_test_encoded, predictions)
precision = precision_score(y_test_encoded, predictions)
recall = recall_score(y_test_encoded, predictions)
f1 = f1_score(y_test_encoded, predictions)

# Extract TP, TN, FP, FN
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

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
print(classification_report(y_test_encoded, predictions))
