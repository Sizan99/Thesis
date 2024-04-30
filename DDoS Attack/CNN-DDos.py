import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

# Pad sequences
max_seq_length = X_train_scaled.shape[1]
X_train_pad = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_pad = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Construct CNN model
model = Sequential()
model.add(Conv1D(64, 5, activation='relu', input_shape=(max_seq_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train_pad, y_train_encoded, validation_data=(X_test_pad, y_test_encoded), epochs=10, batch_size=64)

# Evaluate model
loss, accuracy = model.evaluate(X_test_pad, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = (model.predict(X_test_pad) > 0.5).astype("int32")

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

#ROC Curve
fpr, tpr, _ = roc_curve(y_test_encoded, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
