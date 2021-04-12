'''CATS'''

'''Install/Import packages'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Set work directory'''
# # Use Ctrl+/ to un-/comment:
os.getcwd()
os.chdir('/home/cristanteam/pycodes/VU/B4TM/CATS_project')

'''Import data'''
dataset = pd.read_table('Train_call.txt')
labels = pd.read_table('Train_clinical.txt')

dataset.head()
labels.head()

## Checking for class imbalance
labels['Subgroup'].value_counts()       # data is balanced!
labels[['Sample', 'Subgroup']].value_counts().describe()        # every sample (gene) has 1 subgroup case registered?

## Checking for missing values
dataset.isna().sum()
labels.isna().sum()     # No missing values!

## Inspecting data type
dataset.dtypes      # all encoded as int64
labels.dtypes       # still encoded as object, should be 'category' with 3 levels?
labels.Subgroup.astype('category').cat.set_categories(['HR+', 'HER2+', 'Triple Neg']).cat.codes
# If we apply the code above 'HR+' = 0, 'HER2+' = 1, 'Triple Neg' = 2

## Checking variables names and indexing
dataset.columns
dataset.values
dataset.values[0][4:]
dataset.values[-1][4:]
dataset.loc[0]
dataset.loc[0:, 'Chromosome']       # works with 'string' indexes
dataset.iloc[0]
dataset.iloc[0:, 4:]        # works with 'integer' indexes

## Collecting features
type(dataset)
dataset_arrays = dataset.iloc[0:, 4:]       # all features maintained while excluding 'Chromosome', 'Start', 'End', 'Nclone' columns
dataset_arrays.transpose()
dataset_arrays.T        # '.transpose()' or '.T' to transpose


'''Split data'''
from sklearn.model_selection import train_test_split
train, validation, \
train_labels, validation_labels = \
    train_test_split(dataset_arrays.T, labels,     # attention: '.T' was used
                     test_size=.2,     # save 20% for validation set
                     random_state=57)      # 'random_state=' used as 'seed'

type(train_labels)
train_labels['Subgroup'].value_counts()
validation_labels['Subgroup'].value_counts()        # Still looks balanced, but HER2+ with fewer observations

## Visualizing counts
sns.countplot(train_labels['Subgroup'], label='Counts',
              palette='BuPu',
              order=['HR+', 'HER2+', 'Triple Neg'])
plt.show()
sns.countplot(validation_labels['Subgroup'], label='Counts',
              palette='BuPu',
              order=['HR+', 'HER2+', 'Triple Neg'])
plt.show()

# Random forest classifier #### Reference: https://intellipaat.com/blog/what-is-random-forest-algorithm-in-python/#Random-Forest-Example

## Selecting features and target
X = pd.DataFrame(dataset.iloc[:,:-1])  # pd.DataFrame(dataset.iloc[:, [0,1,3]])
X   # bed_time, stress_level and programme as features
X.programme = X.programme.cat.codes   # X.gender = X.gender.cat.codes   # programmes turned into integer identifiers
y = pd.DataFrame(dataset.iloc[:,-1])    # pd.DataFrame(dataset.iloc[:,-2])
y   # gender, our target
# y.programme = y.programme.cat.codes    # y.gender = y.gender.cat.codes    # 0 = female, 1 = male

## Splitting data into sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

## Generation ensemble random forest model
rf_classifier = RandomForestClassifier(n_estimators=20, criterion='gini',    # Gini impurity to measure split quality
                                    # skipna=True,
                                    max_depth=None, random_state=1)   # 'max_deph=None' to perform split until all leaves are pure
rf_classifier.fit(X_train, y_train)
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(rf_classifier.score(X_train, y_train)), '\n')

## Prediction with training set
y_pred = rf_classifier.predict(X_test)

## Model evaluation
# help(confusion_matrix) # returns array in this order: tn, fp, fn, tp
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred), '\n')   #, labels=[0, 1]))
print('Random Forest classifier metrics:\n\n', classification_report(y_test, y_pred))
print('Accuracy of Random Forest classifier on test set: %.2f \n' % accuracy_score(y_test, y_pred))

## Feature selection
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Feature contributions to prediction:\n')
print(feature_importance, '\n')

sns.barplot(x=feature_importance, y=feature_importance.index,
            # color='darkgreen',
            palette='Greens_r',   # palette options: https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f
            alpha=0.8)
plt.xlabel('Feature importance score')
plt.ylabel('Features')
plt.title('Feature importance using Random Forest classifier')

## Extracting important features
feat_imp = SelectFromModel(rf_classifier, threshold=0.3)   # only features with importance > 0.3 (aka. stress_level solely)
feat_imp.fit(X_train, y_train)

## Pick and store important features in new train and test objects
X_imp_train = feat_imp.transform(X_train)
X_imp_test = feat_imp.transform(X_test)

## Generating new Random Forest classifier with important features
rf_classifier_imp = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=None, random_state=1)
rf_classifier_imp.fit(X_imp_train, y_train)

## Comparing results between old and new Random Forest models
y_pred = rf_classifier.predict(X_test)
print('Accuracy of OLD Random Forest classifier on test set: %.2f \n' % accuracy_score(y_test, y_pred))
y_imp_pred = rf_classifier_imp.predict(X_imp_test)
print('Accuracy of NEW Random Forest classifier on test set: %.2f \n' % accuracy_score(y_test, y_imp_pred))
print('Confusion matrix: \n', confusion_matrix(y_test, y_imp_pred), '\n')



