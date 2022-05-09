#%%
# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

from sklearn.model_selection import train_test_split
#%%

# Importing dataset
dataset = pd.read_csv("data.csv")

# Visualizing table
dataset.head()

#%%
# Dataset info (atributtes)
dataset.info()
#### columns "id" and "Unnamed: 32" are not useful -> get rid of the,

#%%
dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)

#%%
# Class distribution
dataset.diagnosis.value_counts(sort=False).plot(kind='bar')
plt.title('Class distribution', fontsize = 15)
plt.xlabel('Classes', fontsize = 15)
plt.xticks(rotation=360)
plt.ylabel('Quantity', fontsize = 15)
plt.show()
print()

#%%
# Prediction tree
X = dataset.drop('diagnosis', axis=1)
Y = dataset ['diagnosis']

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 14)
print("Train dataset shape: ", X_train.shape )
print("Test dataset shape: ", X_test.shape )

#%%
model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=123)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print('Training set accuracy: {:.4f}'.format(model.score(X_train, Y_train) * 100))
print('Test set accuracy: {:.4f}'.format(model.score(X_test, Y_test) * 100))


fig = plt.figure(figsize=(40,20))
fig = tree.plot_tree(model, feature_names=list(X.columns.values),  class_names=['M', 'B'], filled=True, impurity = False)
plt.show()
print()




