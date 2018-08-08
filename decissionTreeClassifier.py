from sklearn import tree
from sklearn.neural_network import MLPClassifier
# RandomForestClassifier
# import sklearn
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# 1
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])
print(prediction)

# 2
clf = MLPClassifier()
clf.fit(X, Y)
prediction = clf.predict([[190, 70, 39]])
print(prediction)

# 3



# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("human")