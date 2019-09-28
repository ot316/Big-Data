import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import graphviz
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy import stats
stats.chisqprob = lambda chisq,k: stats.chi2.sf(chisq,k)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import statsmodels.genmod.families as sm
reviews = pd.read_csv('LasVegasTripAdvisorReviews_noloc.csv')
reviews.head()
# Write your splitting percentages here
first_split = 0.4
second_split = 0.5
train_h, test_and_val_h = train_test_split(reviews, test_size = first_split, random_state = 42)
val_h, test_h = train_test_split(test_and_val_h, test_size = second_split, random_state = 42)
print(len(train_h), len(val_h), len(test_h))
predictors = list(reviews.columns)
target = 'Binary_Review'
target2 = 'Score'
predictors.remove(target)
predictors.remove(target2)
x_train_h, x_val_h, x_test_h = np.array(train_h[predictors]), np.array(val_h[predictors]) ,
np.array(test_h[predictors])
y_train_h, y_val_h, y_test_h = np.array(train_h[target]), np.array(val_h[target]) , np.array(test_h[target])
r1 = tree.DecisionTreeClassifier(max_depth = 2, min_impurity_decrease= 0.01) # Our classification tree
r1 = r1.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r1.predict(x_train_h))) 
r2 = tree.DecisionTreeClassifier(max_depth = 3, min_impurity_decrease= 0.01) # Our classification tree
r2 = r2.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r2.predict(x_train_h)))
r3 = tree.DecisionTreeClassifier(max_depth = 4, min_impurity_decrease= 0.01) # Our classification tree
r3 = r3.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r3.predict(x_train_h)))
r4 = tree.DecisionTreeClassifier(max_depth = 2, min_impurity_decrease= 0.05) # Our classification tree
r4 = r4.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r4.predict(x_train_h)))
r5 = tree.DecisionTreeClassifier(max_depth = 3, min_impurity_decrease= 0.05) # Our classification tree
r5 = r5.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r5.predict(x_train_h)))
r6 = tree.DecisionTreeClassifier(max_depth = 5, min_impurity_decrease= 0.05) # Our classification tree
r6 = r6.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r6.predict(x_train_h)))
r7 = tree.DecisionTreeClassifier(max_depth = 7, min_impurity_decrease= 0.02) # Our classification tree
r7 = r7.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r7.predict(x_train_h)))
r8 = tree.DecisionTreeClassifier(max_depth = 7, min_impurity_decrease= 0.005) # Our classification tree
r8 = r8.fit(x_train_h, y_train_h)
print('1. Train set accuracy: %.3f'%accuracy_score(y_train_h,r8.predict(x_train_h)))
print('\nFor the training set:')
print('Accuracy: %0.4f'%(accuracy_score(y_train_h, r3.predict(x_train_h))))
print('Precision: %0.4f'%(precision_score(y_train_h, r3.predict(x_train_h))))
print('Recall: %0.4f'%(recall_score(y_train_h, r3.predict(x_train_h))))
print('\nFor the validation set:')
print('Accuracy: %0.4f'%(accuracy_score(y_val_h, r3.predict(x_val_h))))
print('Precision: %0.4f'%(precision_score(y_val_h, r3.predict(x_val_h))))
print('Recall: %0.4f'%(recall_score(y_val_h, r3.predict(x_val_h))))
print('\nFor the test set:')
print('Accuracy: %0.4f'%(accuracy_score(y_test_h, r3.predict(x_test_h))))
print('Precision: %0.4f'%(precision_score(y_test_h, r3.predict(x_test_h))))
print('Recall: %0.4f'%(recall_score(y_test_h, r3.predict(x_test_h))))
dot_data = tree.export_graphviz(r8, out_file=None)
graph = graphviz.Source(dot_data)
dot_data = tree.export_graphviz(r8, out_file=None,
 feature_names = predictors,
 class_names = ('Not 5 stars', '5 Stars'), 
 filled = True, rounded = True,
 special_characters = True)
graph = graphviz.Source(dot_data)
graph 