import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "diabetes.csv"
df = pd.read_csv(file_path)


# Custom Standard Scaler
def standard_scaler(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev


# Scale the features using the custom function
X = df.drop('Outcome', axis=1)
X_scaled_custom = standard_scaler(X)

# Scale the features using scikit-learn's StandardScaler
scaler = StandardScaler()
X_scaled_sklearn = scaler.fit_transform(X)

# Print the means and standard deviations for both scalers
print("Means (Custom Scaler):\n", np.mean(X_scaled_custom, axis=0))
print("Standard Deviations (Custom Scaler):\n", np.std(X_scaled_custom, axis=0))
print("\nMeans (Scikit-learn Scaler):\n", np.mean(X_scaled_sklearn, axis=0))
print("Standard Deviations (Scikit-learn Scaler):\n", np.std(X_scaled_sklearn, axis=0))

# Split the dataset into training and testing
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_scaled_custom, y, test_size=0.2, random_state=42)

# Determine the K Value and Create a Visualization of the Accuracy
k_values = range(1, 21)
accuracy_values = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

# Report the best K value
best_k = k_values[np.argmax(accuracy_values)]
print("\nBest K Value:", best_k)

# Visualize the accuracy
plt.plot(k_values, accuracy_values, label='Accuracy')
plt.axvline(x=best_k, color='red', linestyle='--', label='Best K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('K Value vs. Accuracy')
plt.legend()
plt.show()

# Run 5-Fold Cross Validations - Report Mean and Standard Deviation
knn = KNeighborsClassifier(n_neighbors=best_k)
cv_scores = cross_val_score(knn, X_scaled_custom, y, cv=5)
print("\nMean Cross Validation Score:", np.mean(cv_scores))
print("Standard Deviation of Cross Validation Score:", np.std(cv_scores))

# Evaluate using Confusion Matrix
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Retrain using Leave-One-Out Cross Validation - Report Mean and Standard Deviation
loo = LeaveOneOut()
loo_scores = cross_val_score(knn, X_scaled_custom, y, cv=loo)
print("\nMean LOO Score:", np.mean(loo_scores))
print("Standard Deviation of LOO Score:", np.std(loo_scores))
