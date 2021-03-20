# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from IPython.display import display
# %%
# Zadanie 1.
# Powtórz przykład ze zbiorem Iris, ale podziel dane na treningowe i testowe (np. 80% i 20%). Wykorzystaj
# from sklearn.model_selection import train_test_split
# Naucz model i sprawdź jego skuteczność na zbiorze testowym przy pomocy funkcji/metod z sklearn.metrics
# confusion_matrix
# accuracy_score
# classification_report
iris = sns.load_dataset("iris")
display(iris)
# %%
X = iris.drop('species', axis=1)
y = iris['species']
feature_names = X.columns.tolist()
print(feature_names)
# %%
weight = pd.read_excel('weight.xlsx')
display(weight)
# %%
X = weight.drop('Gender', axis=1)
y = weight['Gender']
feature_names = X.columns.tolist()
print(feature_names)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# %%
classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
classifier.fit(X_train, y_train)
# %%
text_representation = export_text(classifier, feature_names=feature_names)
print(text_representation)
# %%
data = export_graphviz(classifier , out_file=None, feature_names=feature_names)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('decisiontree.png')
# %%
img = pltimg.imread('decisiontree.png')
plt.imshow(img)
plt.axis('off')
plt.show()
# %%
y_pred = classifier.predict(X_test)
# %%
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: '+'{:.2%}'.format(accuracy))
print(classification_report(y_test, y_pred))