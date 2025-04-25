import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.utils import to_categorical
import os

class Enviorement:

    def __init__(self, location, train_file_name, test_file_name, learner_model):
        self.data = pd.read_csv(location + '/' + train_file_name)
        self.test_data = pd.read_csv(location + '/' + test_file_name)
        self.action_space = np.zeros(self.data.shape[1] - 1)
        self.data_size = len(self.data)
        self.learner_model = learner_model
        self.number_of_classes = self.data.iloc[:, -1].unique().shape[0]

    # Function that create the episode data - sample randomaly (need to add boosting)
    def get_data(self, episode_size, for_episode, mode):
        if mode == 'train':
            if not hasattr(self, 'data'):
                self.data = pd.read_csv(os.path.join(self.location, self.train_file_name))

            n_samples = len(self.data)
            actual_sample_size = min(episode_size, n_samples)
            if actual_sample_size < episode_size:
                print(f"Warning: Requested episode_size ({episode_size}) is larger than available training samples ({n_samples}). Using {actual_sample_size} samples.")

            dataset = self.data.sample(n=actual_sample_size, random_state=np.random.randint(1,10000))

        elif mode == 'test':
            if not hasattr(self, 'test_data'):
                self.test_data = pd.read_csv(os.path.join(self.location, self.test_file_name))
            dataset = self.test_data
        else:
            raise ValueError(f"Invalid mode specified: {mode}")

        return dataset

    def data_separate(self, dataset):
        global X
        global y
        X = dataset.iloc[:, 0:dataset.shape[1] - 1]  # all rows, all the features and no labels
        y = dataset.iloc[:, -1]  # all rows, label only
        return X, y

    # Function that split the episode data into train and test
    def data_split(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train_main, X_test_main, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=4)
        return X_train_main, X_test_main, y_train, y_test

    def s2(self, s, a, selected_actions):
        s2 = to_categorical(a, len(self.action_space))
        sel_actions = selected_actions.copy()
        sel_actions[a] = 1
        return s2, sel_actions

    def accuracy(self, s2, X_train_main, X_test_main, y_train, y_test):
        columns = np.where(s2 == 1)[0]
        X_train = X_train_main.iloc[:, columns]
        X_test = X_test_main.iloc[:, columns]
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        accuracy = self.leraner(X_train, X_test, y_train, y_test)
        return accuracy

    def leraner(self, X_train, X_test, y_train, y_test):
        if self.learner_model == 'DT':
            learner = tree.DecisionTreeClassifier()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        elif self.learner_model == 'NB':
            learner = MultinomialNB()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return accuracy