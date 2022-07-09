import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


class Layer:
    def __init__(self, n_estimators, num_forests, num_labels, step=3, layer_index=0, fold=0):
        self.n_estimators = n_estimators
        self.num_labels = num_labels
        self.num_forests = num_forests
        self.layer_index = layer_index
        self.fold = fold
        self.step = step
        self.model = []

    def train(self, train_data, train_label):
        #print("train_data.shape=",train_data.shape)
        n_estimators = min(20 * self.layer_index + self.n_estimators, 600)
        max_depth = self.step * self.layer_index + self.step
        for forest_index in range(self.num_forests):
            if forest_index % 2 == 0:
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion="gini",
                                             max_depth=max_depth,
                                             n_jobs=-1)
            else:
                clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                           criterion="gini",
                                           max_depth=max_depth,
                                           n_jobs=-1)
            train_data[np.isnan(train_data)] = 0
            clf.fit(train_data, train_label)
            self.model.append(clf)
        self.layer_index += 1

    def predict(self, test_data):
        predict_prob = np.zeros(
            [self.num_forests, test_data.shape[0], self.num_labels])

        for forest_index, clf in enumerate(self.model):
            test_data[np.isnan(test_data)] = 0
            predict_p = clf.predict_proba(test_data)
            for j in range(len(predict_p)):
                predict_prob[forest_index, :, j] = 1 - predict_p[j][:, 0].T

        prob_avg = np.sum(predict_prob, axis=0)
        prob_avg /= self.num_forests
        prob_concatenate = predict_prob
        return [prob_avg, prob_concatenate]

    def train_and_predict(self, train_data, train_label, val_data, test_data):
        self.train(train_data, train_label)
        val_avg, val_concatenate = self.predict(val_data)
        prob_avg, prob_concatenate = self.predict(test_data)

        return [val_avg, val_concatenate, prob_avg, prob_concatenate]
