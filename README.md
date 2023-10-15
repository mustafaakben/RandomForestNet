## Overview
`RandomForestNet` is a custom implementation of a Random Forest classifier that utilizes decision stumps built with a simple neural network model in TensorFlow. This approach combines tree-based ensemble learning with deep learning methods, aiming to benefit from both worlds: the high interpretability of decision trees and the capability of neural networks to handle non-linear patterns. 

## Requirements
- Python
- NumPy
- TensorFlow
- scikit-learn
- tqdm

## Quick Start
Ensure you have all the necessary libraries installed:
```bash
pip install numpy tensorflow scikit-learn tqdm
```

Then, you can utilize `RandomForestNet` in your Python scripts as follows:

```python
from RandomForestNet import RandomForestNet
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Creating synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestNet
rf_net = RandomForestNet(n_trees=10, max_col_size=5, bootstrap_percent=0.8, n_col_sample=5)
rf_net.fit(X_train, y_train)

# Predict and evaluate
predictions = rf_net.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, predictions))
```


## Documentation

### Class: Stump
A single decision stump utilizing a shallow neural network for binary classification.

#### Methods
- `build_model(in_shape=1)`: Constructs a shallow neural network model.
- `fit(X, y, features_idx)`: Fits the stump model to data by training over specified feature indices.
- `predict(X)`: Predicts classes for given input features.

### Class: RandomForestNet
An ensemble classifier that employs multiple stumps (trees).

#### Methods
- `combinational_sampling(n_col_sample=100, max_col_size=1, min_col_size=1, n_features=5)`: Generates a sample of unique combinations of feature indices.
- `fit(X, y)`: Fits the random forest model to data.
- `predict(X, probs=False, all_scores=False)`: Predicts classes for given input features.

#### Parameters
- `n_trees`: Number of trees (stumps) to be used in the ensemble.
- `max_col_size`: Maximum number of columns (features) to be sampled for each stump.
- `bootstrap_percent`: Percentage of data to be sampled for training each tree.
- `n_col_sample`: Number of feature index combinations to be sampled for each tree.

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License
The code in this repository is provided under the MIT License.

