from pathlib import Path
import os, sys
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

"""
PART 2 - Text Classification Task with NB, MLP, SVM
"""
# load data from text
BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
with open(BASE_DIR / 'clean_fake.txt', 'r') as f:
    fake = f.read().split('\n')

with open(BASE_DIR / 'clean_real.txt', 'r') as f:
    real = f.read().split('\n')

"""
Extract features from text files
"""
count_vect = CountVectorizer()

y = np.append(np.ones(np.shape(fake)), np.zeros(np.shape(real)))
X = fake + real

X_tf = count_vect.fit_transform(X)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_tf)
X_tf = tf_transformer.transform(X_tf)

X_train, X_test, y_train, y_test = train_test_split(
    X_tf, y, random_state=42, stratify=y, test_size=0.3
)

"""
Train a Narive Bayes classfier
"""
nb = MultinomialNB(alpha=0.001)
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
accuracy1 = np.mean(pred == y_test)
print(f"Classifier {nb} accuracy = {accuracy1}\n")


"""
Train a MLP classfier
"""
# a Multilayer perceptron with 65 hidden neurons
mlp = MLPClassifier(
    hidden_layer_sizes=(65,),
    max_iter=100,
    alpha=1e-4,
    solver="sgd",
    random_state=1,
    learning_rate_init=0.2,
)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
accuracy2 = np.mean(pred == y_test)
print(f"Classifier {mlp} accuracy = {accuracy2}\n")

"""
Building a pipeline of vectorizer => transformer => NB classifier
"""
# regenerate unpreprocessed train / test data for input of pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.3
)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mlp', MultinomialNB()),
])
text_clf.fit(X_train, y_train)
pred = text_clf.predict(X_test)
accuracy3 = np.mean(pred == y_test)
print(f"Pipeline {text_clf} accuracy = {accuracy3}\n")


"""
a pipeline of vectorizer => transformer => SVM classifier
"""
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)),
])
text_clf.fit(X_train, y_train)
pred = text_clf.predict(X_test)
accuracy4 = np.mean(pred == y_test)
print(f"Pipeline {text_clf} accuracy = {accuracy4}\n")


"""
PART 3 - Improve NB for Text Classification on Newsgroup dataset. Use Grid Search to improve performance.
"""
# load dataset
X, y = datasets.fetch_20newsgroups(subset="all", return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.3
)
# build NB pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (0.002, 0.003),
}
# Parameter tuning using grid search
print("Start Grid Search")
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
pred = gs_clf.predict(X_test)
accuracy5 = np.mean(pred == y_test)
print(f"Pipeline from Grid Search {gs_clf} accuracy = {accuracy5}\n")