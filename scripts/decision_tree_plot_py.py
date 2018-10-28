import pydotplus

from sklearn import datasets, model_selection, tree

def main():
    iris = datasets.load_iris()
    X = iris["data"]
    y = iris["target"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    tree_clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tree_clf.fit(X_train,
                 y_train)

    dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                                    feature_names=iris["feature_names"],  
                         class_names=iris["target_names"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png('decision_tree.png')

if __name__ == '__main__':
    main()
