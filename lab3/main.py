import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
CLEAN = 1

def data_preprocessing(data):
    res = []
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for item in data:
        item = word_tokenize(item)
        item = [w for w in item if w not in stop_words]
        item = [lemmatizer.lemmatize(w) for w in item]
        res.append(' '.join(item))
    return res

def print_info(real, predicted):
    print("Матрица ошибок:")
    matrix = confusion_matrix(real, predicted)
    print(matrix)
    acc = accuracy_score(real, predicted)
    prec, rec, fscore, support = precision_recall_fscore_support(real, predicted, average='macro')
    print("Accuracy:", acc) # доля правильных ответов
    print("Precision:", prec) # точность, tp/tp+fp
    print("Recall:", rec) # полнота, tp/tp+fn
    print("F-measure:", fscore)

train = np.genfromtxt('../prep_train.csv', delimiter=',', encoding='utf8', dtype=None)
train = train[1:]
test = np.genfromtxt('../prep_test.csv', delimiter=',', encoding='utf8', dtype=None)
test = test[1:]
train = np.transpose(train)
test = np.transpose(test)

[train_classes], [train_titles], [train_descriptions] = np.split(train, 3)
[test_classes], [test_titles], [test_descriptions] = np.split(test, 3)
if CLEAN:
    train_titles = data_preprocessing(train_titles)
    train_descriptions = data_preprocessing(train_descriptions)
    test_titles = data_preprocessing(test_titles)
    test_descriptions = data_preprocessing(test_descriptions)

tvectorizer = TfidfVectorizer()
train = tvectorizer.fit_transform(train_titles)

knc_title = KNeighborsClassifier()
knc_title.fit(train.toarray(), train_classes)

test = tvectorizer.transform(test_titles)
title_res = knc_title.predict(test.toarray())

print("Анализ заголовков:")
print_info(test_classes, title_res)
print()

dvectorizer = TfidfVectorizer()
train = dvectorizer.fit_transform(train_descriptions)

knc_descr = KNeighborsClassifier()
knc_descr.fit(train.toarray(), train_classes)

test = dvectorizer.transform(test_descriptions)
descr_res = knc_descr.predict(test.toarray())

print("Анализ описаний:")
print_info(test_classes, descr_res)
'''
# Дерево решений
print('tree')
train = tvectorizer.fit_transform(train_titles)

dtc_title = DecisionTreeClassifier()
dtc_title.fit(train.toarray(), train_classes)
print('trained')
test = tvectorizer.transform(test_titles)
title_res = dtc_title.predict(test.toarray())

print("Анализ заголовков:")
print_info(test_classes, title_res)
print()

train = dvectorizer.fit_transform(train_descriptions)

dtc_descr = DecisionTreeClassifier()
dtc_descr.fit(train.toarray(), train_classes)

test = dvectorizer.transform(test_descriptions)
descr_res = dtc_descr.predict(test.toarray())

print("Анализ описаний:")
print_info(test_classes, descr_res)
'''