from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()

X = data['data']
y = data['target']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=44)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_y)

pred_y = clf.predict(test_X)

print(data['target_names'][pred_y])

accuracy = accuracy_score(test_y, pred_y)
print(accuracy)
print(accuracy_score(train_y, clf.predict(train_X)))

