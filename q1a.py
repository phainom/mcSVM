import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import maxabs_scale
from sklearn.svm import SVC


class OneVsAllClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator):
        self.estimator = estimator
        self._estimators = []

    def fit(self, X, y):
        self._classes = np.unique(y)
        y_bin = self._binarize_labels(y)
        for i in range(len(self._classes)):
            estimator = clone(self.estimator)
            self._estimators.append(estimator.fit(X, y_bin[:, i]))
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        max_margin = np.ndarray(n_samples, dtype=float)
        max_margin.fill(-np.inf)
        prediction_idx = np.zeros(n_samples, dtype=int)
        for idx, estimator in enumerate(self._estimators):
            margin = estimator.decision_function(X)
            np.maximum(max_margin, margin, out=max_margin)
            prediction_idx[max_margin == margin] = idx
        return self._classes[prediction_idx]

    def _binarize_labels(self, y):
        binary_labels = np.ndarray((len(y), len(self._classes)), dtype=int)
        binary_labels.fill(-1)
        for idx, label in enumerate(y):
            class_index = np.where(self._classes == label)
            binary_labels[idx, class_index] = 1
        return binary_labels


def run():
    header_names = ('id', 'refractive_index', 'sodium', 'magnesium', 'aluminum',
                    'silicon', 'potassium', 'calcium', 'barium', 'iron', 'y')
    df = pd.read_csv('glass.data', header=None, names=header_names, index_col=0)
    y = df.pop('y')
    X = df
    # scale data to [-1, 1]
    X_scaled = maxabs_scale(X)
    my_clf = OneVsAllClassifier(SVC(kernel='linear', probability=True))
    sk_clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    cv = KFold(X_scaled.shape[0], n_folds=5, shuffle=True)
    score = cross_val_score(sk_clf, X_scaled, y, cv=cv)
    my_score = cross_val_score(my_clf, X_scaled, y, cv=cv)
    print 'Scikit-learns OVR cross-val score: {}'.format(np.mean(score))
    print 'My OVA cross-val score: {}'.format(np.mean(my_score))

if __name__ == '__main__':
    run()
