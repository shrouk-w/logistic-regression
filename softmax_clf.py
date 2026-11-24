import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plotka import plot_decision_regions
from reglog import LogisticRegressionGD


class SoftmaxClassifier(object):
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        self.classifiers_ = []
        for current_class in self.classes_:
            binary_targets = np.where(labels == current_class, 1, 0)
            classifier = LogisticRegressionGD(
                eta=self.eta,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            classifier.fit(features, binary_targets)
            self.classifiers_.append(classifier)
        return self

    def _logits(self, features):
        return np.column_stack([model.net_input(features) for model in self.classifiers_])

    def predict_proba(self, features):
        features = np.atleast_2d(features)
        logits = self._logits(features)
        logits = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, features):
        features = np.atleast_2d(features)
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def print_proba(self, features, true_labels=None):
        features = np.atleast_2d(features)
        probabilities = self.predict_proba(features)
        predicted_labels = self.predict(features)
        header = "Probka | " + " | ".join(
            f"P(y={class_label})" for class_label in self.classes_
        ) + " | pred"
        if true_labels is not None:
            header += " | y_true"
        print(header)
        print("-" * len(header))
        for sample_index, sample_probabilities in enumerate(probabilities):
            row = f"  {sample_index:3d}  | " + " | ".join(f"{probability:7.4f}" for probability in sample_probabilities)
            row += f" |  {predicted_labels[sample_index]}"
            if true_labels is not None:
                row += f"   |   {true_labels[sample_index]}"
            print(row)


def main():
    iris_dataset = datasets.load_iris()
    petal_features = iris_dataset.data[:, [2, 3]]
    target_labels = iris_dataset.target

    train_features, test_features, train_labels, test_labels = train_test_split(
        petal_features, target_labels, test_size=0.3, random_state=1, stratify=target_labels
    )

    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    test_features_std = scaler.transform(test_features)

    softmax_model = SoftmaxClassifier(eta=0.05, n_iter=2000, random_state=1)
    softmax_model.fit(train_features_std, train_labels)

    train_accuracy = np.mean(softmax_model.predict(train_features_std) == train_labels)
    test_accuracy = np.mean(softmax_model.predict(test_features_std) == test_labels)
    print(f"Dokladnosc (train): {train_accuracy:.4f}")
    print(f"Dokladnosc (test):  {test_accuracy:.4f}\n")

    print("Unormowane prawdopodobienstwa dla pierwszych 10 probek testowych:")
    softmax_model.print_proba(test_features_std[:10], true_labels=test_labels[:10])

    combined_features = np.vstack((train_features_std, test_features_std))
    combined_labels = np.hstack((train_labels, test_labels))
    plot_decision_regions(X=combined_features, y=combined_labels, classifier=softmax_model)
    plt.xlabel(r'$x_1$ (dlugosc platka)')
    plt.ylabel(r'$x_2$ (szerokosc platka)')
    plt.title('SOFTMAX (One-vs-Rest) na zbiorze Iris')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
