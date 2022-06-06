import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


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


data = np.genfromtxt('../data.csv', delimiter=',')
data = data[1:]

x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=2)
pca.fit(x_train, y_train)
#x_train = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

plt.figure(1)
plt.title("Реальные классы")
plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test, cmap='Set1')

# Kneighbors
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
knc_res = knc.predict(x_test)
print("K Neighbors:")
print_info(y_test, knc_res)

plt.figure(2)
plt.title("K Neighbors")
plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=knc_res, cmap='Set1')

# Дерево решений
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_res = dtc.predict(x_test)
print("\nДерево решений:")
print_info(y_test, dtc_res)

plt.figure(3)
plt.title("Дерево решений")
plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=dtc_res, cmap='Set1')

plt.show()
