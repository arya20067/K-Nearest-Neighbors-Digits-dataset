# K-Nearest Neighbors on Digits Dataset

This project demonstrates a simple K-Nearest Neighbors (KNN) classifier using the classic digits dataset from scikit-learn. The digits dataset consists of 8x8 images of handwritten digits (0–9).

## Requirements

- Python 3.7+
- scikit-learn
- numpy

Install dependencies with:

```bash
pip install scikit-learn numpy
```

## Usage

Run the main code file:

```bash
python knn_digits.py
```

This will:
- Load the digits dataset
- Split the data into training and test sets
- Train a KNN classifier (k=5)
- Output accuracy and a classification report

## About the Code

- Uses `KNeighborsClassifier` from scikit-learn.
- Splits the dataset with a 70/30 train/test split.
- Prints accuracy and detailed per-class metrics.

## Customization

You can change the number of neighbors (k) by editing:

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

## License

This project is for educational use.