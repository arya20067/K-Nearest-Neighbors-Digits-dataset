import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load the digits dataset
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Print results
    print("K-Nearest Neighbors Classifier on Digits Dataset")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()