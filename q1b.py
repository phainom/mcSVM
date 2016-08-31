import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import maxabs_scale
from sklearn.svm import SVC


class OneVsOne(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator):
        self.estimator = estimator
        self._estimators = []

    def fit(self, X, y):
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_one = self._classes[i]
                class_two = self._classes[j]
                two_classes = np.logical_or(y == class_one, y == class_two)
                subset_of_y = y[two_classes]
                y_bin = np.zeros(subset_of_y.shape, dtype=int)
                # class_one is zero
                y_bin[subset_of_y == class_two] = 1
                estimator = clone(self.estimator)
                self._estimators.append((estimator.fit(X[two_classes], y_bin)))
        return self

    def predict(self, X):
        predictions = [estimator.predict(X) for estimator in self._estimators]
        predictions = np.asarray(predictions).T
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        votes = np.zeros((n_samples, n_classes), dtype=int)
        column = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                votes[predictions[:, column] == 0, i] += 1
                votes[predictions[:, column] == 1, j] += 1
                column += 1
        class_idx = votes.argmax(axis=1)
        return self._classes[class_idx]


def run():
    header_names = ('id', 'refractive_index', 'sodium', 'magnesium', 'aluminum',
                    'silicon', 'potassium', 'calcium', 'barium', 'iron', 'y')
    df = pd.read_csv('glass.data', header=None, names=header_names, index_col=0)
    y = df.pop('y')
    X = df
    # scale data to [-1, 1]
    X_scaled = maxabs_scale(X)

    sk_clf = OneVsOneClassifier(SVC(kernel='linear'))
    my_clf = OneVsOne(SVC(kernel='linear', probability=True))
    cv = KFold(X_scaled.shape[0], n_folds=5, shuffle=True)
    score = cross_val_score(sk_clf, X_scaled, y.values, cv=cv)
    my_score = cross_val_score(my_clf, X_scaled, y.values, cv=cv)
    print 'Scikit-learns OVO cross-val score: {}'.format(np.mean(score))
    print 'My OVO cross-val score: {}'.format(np.mean(my_score))

if __name__ == '__main__':
    run()
