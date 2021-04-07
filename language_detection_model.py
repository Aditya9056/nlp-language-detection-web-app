# Importing Libraries
import string
import re
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import pickle

# Data Loading
eng_df = pd.read_csv(
    "languages_dataset/eng_data.csv", encoding="utf-8", header=None, names=["English"]
)

fr_df = pd.read_csv(
    "languages_dataset/fr_data.csv", encoding="utf-8", header=None, names=["French"]
)

chi_df = pd.read_csv(
    "languages_dataset/chi_data.csv", encoding="utf-8", header=None, names=["Chinese"]
)

es_df = pd.read_csv(
    "languages_dataset/es_data.csv", encoding="utf-8", header=None, names=["Spanish"]
)

ru_df = pd.read_csv(
    "languages_dataset/ru_data.csv", encoding="utf-8", header=None, names=["Russian"]
)

eng_df.head()
fr_df.head()
chi_df.head()
es_df.head()
ru_df.head()

# Removing all punctuation, as it is not adding useful information
# for char in string.punctuation:
# print(char, end=" ")
translate_table = dict((ord(char), None) for char in string.punctuation)

data_eng = []
lang_eng = []

data_fr = []
lang_fr = []

data_chi = []
lang_chi = []

data_es = []
lang_es = []

data_ru = []
lang_ru = []

for i, line in eng_df.iterrows():
    line = line["English"]
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"d+", "", line)
        line = line.translate(translate_table)
        data_eng.append(line)
        lang_eng.append("English")

for i, line in fr_df.iterrows():
    line = line["French"]
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"d+", "", line)
        line = line.translate(translate_table)
        data_fr.append(line)
        lang_fr.append("French")

for i, line in chi_df.iterrows():
    line = line["Chinese"]
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"d+", "", line)
        line = re.sub(r"[a-zA-Z]+", "", line)
        line = line.translate(translate_table)
        data_chi.append(line)
        lang_chi.append("Chinese")

for i, line in es_df.iterrows():
    line = line["Spanish"]
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"d+", "", line)
        line = line.translate(translate_table)
        data_es.append(line)
        lang_es.append("Spanish")

for i, line in ru_df.iterrows():
    line = line["Russian"]
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"d+", "", line)
        line = line.translate(translate_table)
        data_ru.append(line)
        lang_ru.append("Russian")

df = pd.DataFrame(
    {
        "Text": data_eng + data_fr + data_chi + data_es + data_ru,
        "language": lang_eng + lang_fr + lang_chi + lang_es + lang_ru,
    }
)

# Export to check file
# df.to_csv("data.csv")

# Splitting Data into Train and Tests sets. {random_state=0 means that same part will go the model without getting shuffled}
X, y = df.iloc[:, 0], df.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Vectorization
vectorizer = feature_extraction.text.TfidfVectorizer(
    ngram_range=(1, 3), analyzer="char"
)

pipe_lr_r13 = pipeline.Pipeline(
    [("vectorizer", vectorizer), ("clf", linear_model.LogisticRegression())]
)

# Model Fitting
pipe_lr_r13.fit(X_train, y_train)


# Persist modelso that it can be used by different consumers
lrFile = open("LRModel.pckl", "wb")
pickle.dump(pipe_lr_r13, lrFile)
lrFile.close()

# Model Loading
global lrLangDetectModel
lrLangDetectFile = open("LRModel.pckl", "rb")
lrLangDetectModel = pickle.load(lrLangDetectFile)
lrLangDetectFile.close()

# Model Prediction
y_predicted = pipe_lr_r13.predict(X_test)

# Model Evaluation
acc = (metrics.accuracy_score(y_test, y_predicted)) * 100
print(acc, "%")

matrix = metrics.confusion_matrix(y_test, y_predicted)
print("Confusion matrix: \n", matrix)