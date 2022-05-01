# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# load data_set, split
X, y = fetch_openml(name="mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

"""
Naive Bayes
"""
# experiments Multinomial Naive Bayes with different alpha 1 * 10^(-i) for i from 0 to 10
# it achieves best accuracy when alpha = 10^(-10)
for i in range(0, 11):
    alpha = 10 ** (-i)
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train, y_train)
    predicted = nb.predict(X_test)
    print(f"Classification report for classifier {nb}:\n")
    print(f"Accuracy = {accuracy_score(y_test, predicted)*100} %")

# print detailed report for the last experiment with best accuracy
print(f"{metrics.classification_report(y_test, predicted)}\n")


"""
Multi-layer Perceptron
"""
# a Multilayer perceptron with 50 hidden neurons: 28x28 inputs --> 50 hidden neurons --> 10 outputs (classes)
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=200,
    alpha=.1,
    solver="sgd",
    random_state=1,
    learning_rate_init=0.2
)

# fit
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))


predicted2=mlp.predict(X_test)

print(
    f"Classification report for classifier {mlp}:\n"
    f"{metrics.classification_report(y_test, predicted2)}\n"
)