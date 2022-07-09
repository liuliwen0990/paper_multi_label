import numpy as np
from .Layer import Layer


class KfoldWarpper:
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, step=3):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.step = step
        self.model = []

    def train(self, train_data, train_label):
        self.num_labels = train_label.shape[1]
        #num_samples, a ,num_features = train_data.shape
        num_samples,num_features = train_data.shape
        prob = np.empty([num_samples, self.num_labels])
        prob_concatenate = np.empty([self.num_forests, num_samples, self.num_labels])

        fold = 0
        for train_index, test_index in self.kf.split(train_data):
            #print(train_index.shape,test_index.shape)
            train_data[np.isnan(train_data)] = 0
            X_train = train_data[train_index,:]
            #print(X_train.shape)
            X_val = train_data[test_index,:]
            y_train = train_label[train_index, :]

            # training fold-th layer
            layer = Layer(self.n_estimators, self.num_forests, self.num_labels, self.step, self.layer_index,
                          fold)
            #print("是否有无穷值：", np.isinf(X_train).any())  # False:没有
            #print ("是否有缺省值：", np.isnan(X_train).any())  # False没有


            layer.train(X_train, y_train)

            self.model.append(layer)
            fold += 1
            prob[test_index], prob_concatenate[:, test_index, :] = layer.predict(X_val)
            
        return [prob, prob_concatenate]

    def predict(self, test_data):

        test_prob = np.zeros([test_data.shape[0], self.num_labels])
        test_prob_concatenate = np.zeros([self.num_forests, test_data.shape[0], self.num_labels])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = layer.predict(test_data)
            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold
        return [test_prob, test_prob_concatenate]

    def train_and_predict(self, train_data, train_label, test_data):
        prob, prob_concatenate = self.train(train_data, train_label)
        test_prob, test_prob_concatenate = self.predict(test_data)
        return [prob, prob_concatenate, test_prob, test_prob_concatenate]
