import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
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

def reduct_data(data, n=5):
    return data[:int(len(data)/n)]

train = np.genfromtxt('../prep_train.csv', delimiter=',', encoding='utf8', dtype=None)
train = train[1:]
test = np.genfromtxt('../prep_test.csv', delimiter=',', encoding='utf8', dtype=None)
test = test[1:]
train = np.transpose(train)
test = np.transpose(test)

[train_classes], [train_titles], [train_descriptions] = np.split(train, 3)
[test_classes], [test_titles], [test_descriptions] = np.split(test, 3)
train_classes = train_classes.astype(int)
test_classes = test_classes.astype(int)

train_classes = reduct_data(train_classes, 2)
test_classes = reduct_data(test_classes, 2)
if CLEAN:
    #train_titles = data_preprocessing(reduct_data(train_titles))
    train_descriptions = data_preprocessing(reduct_data(train_descriptions, 2))
    #test_titles = data_preprocessing(reduct_data(test_titles))
    test_descriptions = data_preprocessing(reduct_data(test_descriptions, 2))


dvectorizer = TfidfVectorizer()
train = dvectorizer.fit_transform(train_descriptions)
test = dvectorizer.transform(test_descriptions)

pca = PCA(n_components=2)
pca.fit(train.toarray(), train_classes)
test_pca = pca.transform(test.toarray())

mlpc = MLPClassifier(hidden_layer_sizes=(8,8), max_iter=500)
mlpc.fit(train.toarray(), train_classes)
test = dvectorizer.transform(test_descriptions)
descr_res = mlpc.predict(test.toarray())
print("Анализ описаний (8):")
print_info(test_classes, descr_res)
print()
plt.figure(1)
plt.title("Реальные классы")
plt.scatter(test_pca[:, 0], test_pca[:, 1], c=test_classes, cmap='Set1')
plt.figure(2)
plt.title("MLPC 8")
plt.scatter(test_pca[:, 0], test_pca[:, 1], c=descr_res, cmap='Set1')

mlpc = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=500)
mlpc.fit(train.toarray(), train_classes)
test = dvectorizer.transform(test_descriptions)
descr_res = mlpc.predict(test.toarray())
print("Анализ описаний (64):")
print_info(test_classes, descr_res)
print()

plt.figure(3)
plt.title("MLPC 64")
plt.scatter(test_pca[:, 0], test_pca[:, 1], c=descr_res, cmap='Set1')
plt.show()
'''
dvectorizer = TfidfVectorizer()
train = dvectorizer.fit_transform(train_descriptions)
arr = [4, 8, 16, 32, 64, 128, 256]
acc = []
for x in arr:
    mlpc = MLPClassifier(hidden_layer_sizes=(x, x), max_iter=500)
    mlpc.fit(train.toarray(), train_classes)
    print('trained')
    test = dvectorizer.transform(test_descriptions)
    descr_res = mlpc.predict(test.toarray())
    acc.append(accuracy_score(test_classes, descr_res))
plt.figure(1)
plt.plot(arr, acc)
acc1 = []
for i in range(len(arr)-1):
    mlpc = MLPClassifier(hidden_layer_sizes=(arr[i+1], arr[i]), max_iter=500)
    mlpc.fit(train.toarray(), train_classes)
    print('trained')
    test = dvectorizer.transform(test_descriptions)
    descr_res = mlpc.predict(test.toarray())
    acc1.append(accuracy_score(test_classes, descr_res))
plt.figure(2)
plt.plot(arr[1:], acc1)

plt.show()
'''