
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

pd.set_option('display.max_columns', None)

# Import movie reviews and flag their respective sentiment


def import_files(path_to_folder, list_name, sentiment):
    path = path_to_folder
    files = glob.glob(path)
    list_name = []
    for txt in files:
        f = open(txt, 'r')
        list_name.append((f.readlines()[0], sentiment))
        f.close()

import_files('./inputs/neg/*.txt', 'negatives_list', 0)



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

# Function to stemm words
stemmer = PorterStemmer()


# Function to clean text
def processStrig(text):
    # process the tweets
    # Convert to lower case
    text = text.lower()
    # replace string: <br /><br />
    text = text.replace('<br /><br />', ' ')
    # Remove stopwords
    text = ' '.join([w for w in text.split() if not w in stopwords.words('english')])
    # strip special characters
    text = re.sub('([^\s\w]|_)+', '', text)
    # Remove leading/trailing blanks
    text = text.strip()
    # Stemm words
    text = ' '.join([stemmer.stem(w) for w in text.split()])
    return text


for i, name in enumerate(features_train):
    features_train[i] = processStrig(features_train[i])

for i, name in enumerate(features_test):
    features_test[i] = processStrig(features_test[i])

# Create train_features, word counts by document - Document-Term matrix - (in sparse format given by CountVectorizer)
# Remove stopwords and select only 1-grams
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=3)
features_train_ = vectorizer.fit_transform(features_train)
features_test_ = vectorizer.transform(features_test)

# Check Document-Term matrix in dense format
pd.DataFrame(features_train_.toarray(), columns=vectorizer.get_feature_names()).shape

# Fit Naive Bayes model
nb = MultinomialNB()
nb.fit(features_train_, target_train)

# Now we can use the model to predict classifications for our test features.
target_predicted = nb.predict_proba(features_test_)[:,1]
# predict class (rather than probabilities) for classification report
target_predicted_report = nb.predict(features_test_)


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
print("Model accuracy is: " + str(accuracy_score(target_test,target_predicted_report)))