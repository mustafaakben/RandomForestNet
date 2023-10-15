import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Stump():
    def __init__(self):
        self.column_index = 0
        self.value = 0
    
    
    def build_model(self, in_shape = 1):
        in_layer = tf.keras.layers.Input(shape=in_shape)
        
        net = tf.keras.layers.Dense(32,activation='relu')(in_layer)
        net = tf.keras.layers.Dropout(0.2)(net)
        out  = tf.keras.layers.Dense(1,activation='sigmoid')(net)
        
        model = tf.keras.Model(inputs=in_layer, outputs=out)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model

    def fit(self, X, y, features_idx):
        min_error = float('inf')
        for idx in features_idx:
            X_temp = X[:, idx]
            
            in_shape = X_temp.shape
            in_shape = in_shape[1:]
            
            if len(in_shape) == 0:
                in_shape = 1
            
            model = self.build_model(in_shape)
            
            call_back = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            
            hist = model.fit(X[:, idx], y, epochs=100, verbose=0,validation_split=.15,callbacks=[call_back])
            
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
    def __init__(self, n_trees = 10, max_col_size = 3,min_col_size=2, bootstrap_percent=0.5,n_col_sample=5):
        
        self.n_trees = n_trees
        self.trees = []
        self.max_col_size = max_col_size 
        self.bootstrap_percent = bootstrap_percent
        self.n_col_sample = n_col_sample
        self.min_col_size = min_col_size

    
    def combinational_sampling(self,
                               n_col_sample=100,
                               max_col_size=2,
                               min_col_size=1,
                               n_features=5):
        feature_range = np.arange(0, n_features)

        if max_col_size is None:
            max_col_size = range(min_col_size, n_features+1)
        else:
            max_col_size = range(min_col_size, max_col_size+1)
        if max(max_col_size) > n_features:
            max_col_size = range(min_col_size, n_features+1)
        
        feature_idx = []
        for _ in range(n_col_sample):
            max_size = np.random.choice(max_col_size, size=1, replace=False)
            feature_idx.append(np.random.choice(feature_range, size=max_size, replace=False).tolist())
        ## Omit Duplicated Feature_idx
        feature_idx = [tuple(lst) for lst in feature_idx]
        # Remove duplicates
        feature_idx = list(set(feature_idx))
        feature_idx = [list(tpl) for tpl in feature_idx]
        return feature_idx



    def fit(self, X, y):
        for _ in tqdm(range(self.n_trees)):
            # Bootstrap sampling
            idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
            X_sample, y_sample = X[idx], y[idx]
            # Random feature selection
            
            features_idx = self.combinational_sampling(n_col_sample=self.n_col_sample,
                                                       max_col_size=self.max_col_size,
                                                       min_col_size=self.min_col_size,
                                                       n_features=X.shape[1])
            # Train stump
            tree = Stump()
            tree.fit(X_sample, y_sample, features_idx)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0)).astype(int)



#########################################################################
#########################################################################
### ANCHOR Load data

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


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
### ANCHOR Train and predict with the RandomForestNet

rfNet = RandomForestNet(n_trees=20, 
                        max_col_size=30,
                        min_col_size=5, 
                        bootstrap_percent=0.5,
                        n_col_sample=5)

rfNet.fit(X_train, y_train)
predictions = rfNet.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
