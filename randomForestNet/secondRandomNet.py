#########################################################################
#########################################################################
# ANCHOR Libraries

import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#########################################################################
#########################################################################
# ANCHOR GPU settings

device_list = tf.config.list_physical_devices()
for i in device_list:
    print(i)

#########################################################################
#########################################################################
# ANCHOR Functions


def clc():
    os.system('cls' if os.name == 'nt' else 'clear')


clc()



class Stump():
    def __init__(self):
        self.column_index = 0
        self.value = 0

    def build_model(self, in_shape=1):
        
        in_layer = tf.keras.layers.Input(shape=in_shape)
        net = tf.keras.layers.Dense(32, activation='relu')(in_layer)
        net = tf.keras.layers.Dropout(0.3)(net)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(net)

        model = tf.keras.Model(inputs=in_layer, outputs=out)
        
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', 
                      metrics=['accuracy'])

        return model

    
    def fit(self, X, y, features_idx):

        min_error = float('inf')

        for idx in tqdm(features_idx):
            X_temp = X[:, idx]

            in_shape = X_temp.shape
            in_shape = in_shape[1:]

            if len(in_shape) == 0:
                in_shape = 1

            in_shape= in_shape[0]
            
            model = self.build_model(in_shape)

            call_back = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            hist = model.fit(X_temp,
                             y,
                             epochs=100,
                             batch_size=100,
                             verbose=0,
                             validation_split=.15,
                             callbacks=[call_back])

            error = np.array(hist.history['val_loss']).min()

            if error < min_error:
                self.column_index = idx
                self.value = model
                self.min_error = error

    def predict(self, X):
        model = self.value
        X_temp = X[:, self.column_index]
        return model.predict(X_temp)


class RandomForestNet():
    def __init__(self,
                 n_trees=10,
                 max_col_size=10,
                 bootstrap_percent=0.8,
                 n_col_sample=5):

        self.n_trees = n_trees
        self.trees = []
        self.max_col_size = max_col_size
        self.bootstrap_percent = bootstrap_percent
        self.n_col_sample = n_col_sample

 
    def combinational_sampling(self, n_col_sample=100, max_col_size=1,min_col_size=1, n_features=5):
        """
        Generate a sample of unique combinations of feature indices.

        Parameters:
        - n_col_sample : int, default=100
            Number of combinations to sample.
        - max_col_size : int, default=1
            Maximum size of each combination.
        - n_features : int, default=5
            Total number of features available to be sampled.

        Returns:
        - feature_idx : list of lists
            A list containing unique combinations of feature indices.
        """
        # Validate and adjust the max_col_size
        max_col_size = min(max_col_size, n_features)
        min_col_size = min(min_col_size, max_col_size)
        
        
        feature_idx = set()

        # Ensure we don't enter an infinite loop if it's impossible to generate desired n_col_sample
        max_attempts = n_col_sample * 10

        attempts = 0
        
        while len(feature_idx) < n_col_sample and attempts < max_attempts:
            max_size = np.random.randint(min_col_size, max_col_size + 1)
            combination = tuple(np.random.choice(n_features, size=max_size, replace=False))
            feature_idx.add(combination)
            attempts += 1

        # Convert tuples back to lists
        return [list(comb) for comb in feature_idx]
    
    

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap sampling
            idx = np.random.choice(np.arange(len(X)), size=int(self.bootstrap_percent*len(X)), replace=True)
            
            X_sample, y_sample = X[idx], y[idx]

            # Random feature selection
            features_idx = self.combinational_sampling(n_col_sample=self.n_col_sample,
                                                       max_col_size=self.max_col_size,
                                                       min_col_size=int(X.shape[1]*.5),
                                                       n_features=X.shape[1])
            # Train stump
            tree = Stump()
            tree.fit(X_sample, y_sample, features_idx)
            
            self.trees.append(tree)
            print("Tree is trained")

    def predict(self, X, probs=False, all_scores=False):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        if all_scores:
            return tree_preds
        if probs:
            return np.mean(tree_preds, axis=0)
        else:
            return np.round(np.mean(tree_preds, axis=0)).astype(int)


#########################################################################
#########################################################################
# ANCHOR Load data for Test

df = datasets.load_breast_cancer()
X, y = df.data, df.target
y = np.array(y == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#########################################################################
#########################################################################
# ANCHOR Let's Test RandomForest from sklearn

rf = RandomForestClassifier(n_estimators=100, max_features=3, bootstrap=True)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(f'Random Forest : Accuracy: {accuracy_score(y_test, predictions)}')

#########################################################################
#########################################################################
# ANCHOR Let's test RandomForestNet


rfNet = RandomForestNet(n_trees=100,
                        max_col_size=30,
                        bootstrap_percent=0.85,
                        n_col_sample=30)

rfNet.fit(X_train, y_train)

predictionsNet = rfNet.predict(X_test, probs=False)
print(f'Random Forest Net Accuracy: {accuracy_score(y_test, predictionsNet)}')


#########################################################################
#########################################################################
# ANCHOR Compare the results

print(f'Random Forest Accuracy: {accuracy_score(y_test, predictions)}')
print(f'Random Forest Net Accuracy: {accuracy_score(y_test, predictionsNet)}')


#########################################################################
#########################################################################
# ANCHOR OLD CODES


def combinational_samplingWhile(self, n_col_sample=100, max_col_size=1, n_features=5):
    """
    Generate a sample of unique combinations of feature indices.

    Parameters:
    - n_col_sample : int, default=100
        Number of combinations to sample.
    - max_col_size : int, default=1
        Maximum size of each combination.
    - n_features : int, default=5
        Total number of features available to be sampled.

    Returns:
    - feature_idx : list of lists
        A list containing unique combinations of feature indices.
    """
    # Validate and adjust the max_col_size
    max_col_size = min(max_col_size, n_features)

    feature_idx = set()

    # Ensure we don't enter an infinite loop if it's impossible to generate desired n_col_sample
    max_attempts = n_col_sample * 10

    attempts = 0
    while len(feature_idx) < n_col_sample and attempts < max_attempts:
        max_size = np.random.randint(1, max_col_size + 1)
        combination = tuple(np.random.choice(
            n_features, size=max_size, replace=False))
        feature_idx.add(combination)
        attempts += 1

    # Convert tuples back to lists
    return [list(comb) for comb in feature_idx]



def combinational_samplingBasic(self, n_col_sample=100, max_col_size=1, n_features=5):
        feature_range = np.arange(0, n_features)

        if max_col_size is None:
            max_col_size = range(1, n_features+1)
        else:
            max_col_size = range(1, max_col_size+1)
        if max(max_col_size) > n_features:
            max_col_size = range(1, n_features+1)

        feature_idx = []
        for _ in range(n_col_sample):
            max_size = np.random.choice(max_col_size, size=1, replace=False)
            feature_idx.append(np.random.choice(
                feature_range, size=max_size, replace=False).tolist())
        # Omit Duplicated Feature_idx
        feature_idx = [tuple(lst) for lst in feature_idx]
        # Remove duplicates
        feature_idx = list(set(feature_idx))
        feature_idx = [list(tpl) for tpl in feature_idx]
        return feature_idx
    
    



def build_model(self, input_dim=5):
    
        net = list()
        for i in range(input_dim):
            net.append(tf.keras.Input(shape=(1,), name=f'feature{i}'))

        secon_embed = list()
        for i in net:
            secon_embed.append(tf.keras.layers.Dense(
                1, activation='linear')(i))

        embeds = list()
        for each_in in net:
            embeds.append(tf.keras.layers.Dense(
                1, activation='sigmoid')(each_in))

        gates = list()
        for each_embeds, in_net in zip(embeds, secon_embed):
            gates.append(each_embeds*in_net)

        # Stack the gates
        stacked = tf.stack(gates, axis=1)
        # Squash the stacked gates
        stacked = tf.keras.layers.Flatten()(stacked)

        out = tf.keras.layers.Dense(1, activation='sigmoid')(stacked)
        model = tf.keras.Model(inputs=net, outputs=out)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model
    



 

    
def combinational_sampling(self, n_col_sample=100, max_col_size=1,min_col_size=1, n_features=5):
    """
    Generate a sample of unique combinations of feature indices.

    Parameters:
    - n_col_sample : int, default=100
        Number of combinations to sample.
    - max_col_size : int, default=1
        Maximum size of each combination.
    - n_features : int, default=5
        Total number of features available to be sampled.

    Returns:
    - feature_idx : list of lists
        A list containing unique combinations of feature indices.
    """
    # Validate and adjust the max_col_size
    max_col_size = min(max_col_size, n_features)
    min_col_size = min(min_col_size, max_col_size)
    
    
    feature_idx = set()

    # Ensure we don't enter an infinite loop if it's impossible to generate desired n_col_sample
    max_attempts = n_col_sample * 10

    attempts = 0
    
    while len(feature_idx) < n_col_sample and attempts < max_attempts:
        max_size = np.random.randint(min_col_size, max_col_size + 1)
        combination = tuple(np.random.choice(n_features, size=max_size, replace=False))
        feature_idx.add(combination)
        attempts += 1

    # Convert tuples back to lists
    return [list(comb) for comb in feature_idx]
 
 