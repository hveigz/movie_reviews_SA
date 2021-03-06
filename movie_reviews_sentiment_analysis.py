
import glob
from time import time
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, confusion_matrix, \
    classification_report

pd.set_option('display.max_columns', None)


# Import movie reviews and flag their respective sentiment

# inputs={1:'negatives_list', 2:'positives_list', 3:'negatives_list_test', 4:'positives_list_test'}


path = './inputs/neg/*.txt'
files = glob.glob(path)
negatives_list = []
for txt in files:
    f = open(txt, 'r')
    negatives_list.append((f.readlines()[0], 0))
    f.close()

path = './inputs/pos/*.txt'
files = glob.glob(path)
positives_list = []
for txt in files:
    f = open(txt, 'r')
    positives_list.append((f.readlines()[0], 1))
    f.close()

path = './inputs/test/*_neg.txt'
files = glob.glob(path)
negatives_list_test = []
for txt in files:
    f = open(txt, 'r')
    negatives_list_test.append((f.readlines()[0], 0))
    f.close()

path = './inputs/test/*_pos.txt'
files = glob.glob(path)
positives_list_test = []
for txt in files:
    f = open(txt, 'r')
    positives_list_test.append((f.readlines()[0], 1))
    f.close()

# Read reviews into lists
features_train = []
target_train = []
for i, name in enumerate(negatives_list + positives_list):
    features_train.append(name[0])
    target_train.append(name[1])

features_test = []
target_test = []
for i, name in enumerate(negatives_list_test + positives_list_test):
    features_test.append(name[0])
    target_test.append(name[1])


# Create class to pre-process string
class processStrig(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        # process the tweets
        # Import stemmer from NLTK
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        # Convert to lower case
        X = map(str.lower, X)
        # Replace string: <br /><br />
        X = map(lambda s: s.replace("<br /><br />", " "), X)
        # Remove stopwords
        X = map(lambda s: ' '.join([w for w in s.split() if not w in stopwords.words('english')]), X)
        # Strip special characters
        X = map(lambda s: re.sub('([^\s\w]|_)+', '', s), X)
        # Remove leading/trailing blanks
        X = map(str.strip, X)
        # Stemm words
        X = map(lambda s: ' '.join([stemmer.stem(w) for w in s.split()]), X)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


# Check Document-Term matrix in dense format
# pd.DataFrame(features_train_.toarray(), columns=vectorizer.get_feature_names()).shape

# Create pipeline to run whole pre-processing and model fitting in one step
pipeline = Pipeline([
    ('str_pre_proc', processStrig()),
    ('ngram', CountVectorizer(ngram_range=(1, 3), min_df=3)),
    ('estimator', MultinomialNB())
    ])

# # Setup the hyperparameter grid
# param_grid = { 'estimator__alpha': [0.2, 0.4, 0.6, 0.8, 1] }
#
# grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1)
#
# start = time()
# fitted_model = grid.fit(features_train, target_train)
# end = time()
# print 'GridSearch fitting time: ' + str(end-start)
#
# grid.best_params_
# grid.best_score_
#
# target_predicted = grid.predict_proba(features_test)[:, 1]


# train the classifier
model = pipeline.fit(features_train, target_train)
# test the classifier
target_predicted = model.predict_proba(features_test)[:, 1]
# predict class (rather than probabilities) for classification report
target_predicted_report = model.predict(features_test)

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(target_test, target_predicted)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Naive bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classifier ROC Curve')
plt.show()

# Compute AUC metric of test set
print("Multinomial naive bayes AUC: {0}".format(roc_auc_score(target_test, target_predicted)))
# compute accuracy
print("Model accuracy is: " + str(accuracy_score(target_test, target_predicted_report)))



# Generate Precision-Recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(target_test, target_predicted)

# Plot Precision-Recall curve
plt.plot(recall, precision, label='Naive bayes Classifier Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Classifier Precision-Recall curve')
plt.show()

print("New model")
print(classification_report(target_test, target_predicted_report))


# Test 2 made-up exmaples
model.predict_proba(["weird movie!!!","Love this movie!"])[:, 1]