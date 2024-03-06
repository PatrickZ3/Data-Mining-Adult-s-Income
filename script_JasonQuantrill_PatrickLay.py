import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ucimlrepo import fetch_ucirepo

import xgboost as xgb

##################
## Load the Dataset

adult = fetch_ucirepo(id=2) 

X = adult.data.features 
y = adult.data.targets 

# Create dataframe from uci
df = X.join(y)


##################
## Show data info

print('\n================== dataframe ==================')
display(df)

print('\n================== metadata ==================')
print(adult.metadata) 
  
print('\n================== variables ==================')
print(adult.variables)

print('\n================== info ==================')
print(df.info())



##################
## Data Cleaning

print('\n================== shape before cleaning ==================')
print(df.shape)

# Change '?' to NA so those rows can be dropped in the next step
df = df.replace('?', pd.NA)

# Drop rows containing NA
df = df.dropna()

print('\n================== shape after dropping NA ==================')
print(df.shape)

# 'education' is a redundant column
# 'education-num' is the ordinally encoded column representing 'education'
df = df.drop('education', axis=1)

# 'income' has different formats for its data
# fix this column to make it consistent
df['income'] = df['income'].replace('>50K.', '>50K')
df['income'] = df['income'].replace('<=50K.', '<=50K')

print('\n================== final shape after cleaning ==================')
print(df.shape)


##################
## Data Preprocessing

# Binarize Target Variable
df['income'] = np.where(df['income'] == '>50K', 1, 0)
df.rename(columns={'income':'income>50K'}, inplace=True)

# Show reformatted target variable
print('\n================== target variable column ==================')
display(df['income>50K'])

# Split features from target
y3 = df.iloc[:,-1:]
X3 = df.iloc[:,1:-1]

# Aggregration - marital-status
X3['marital-status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married', inplace=True)
X3['marital-status'].replace(['Divorced', 'Separated', 'Widowed'], 'Divorced', inplace=True)

# Aggregration - workclass
X3['workclass'].replace(['Federal-gov', 'Local-gov', 'State-gov'], 'government', inplace=True)
X3['workclass'].replace(['Never-worked', 'Without-pay'], 'jobless', inplace=True)

# Aggregration - occupation
X3['occupation'].replace(['Tech-support', 'Craft-repair', 'Machine-op-inspct'], 'Technical/Support', inplace=True)
X3['occupation'].replace(['Other-service', 'Priv-house-serv', 'Protective-serv'], 'Service', inplace=True)
X3['occupation'].replace(['Exec-managerial', 'Adm-clerical'], 'Management/Administration', inplace=True)
X3['occupation'].replace(['Handlers-cleaners', 'Farming-fishing', 'Transport-moving'], 'Manual Labor', inplace=True)

# Aggregration - relationship
X3['relationship'].replace(['Wife', 'Husband'], 'Spouse', inplace=True)
X3['relationship'].replace(['Not-in-family', 'Unmarried'], 'Non-Family', inplace=True)

# Aggregration - native-country
X3['native-country'].replace(['United-States', 'Canada', 'Outlying-US(Guam-USVI-etc)'], 'North America', inplace=True)
X3['native-country'].replace(['England', 'Germany', 'Greece', 'Italy', 'Poland', 'Portugal', 'Ireland', 'France', 'Scotland', 'Yugoslavia', 'Hungary', 'Holand-Netherlands'], 'Europe', inplace=True)
X3['native-country'].replace(['Cambodia', 'India', 'Japan', 'China', 'Philippines', 'Vietnam', 'Taiwan', 'Laos', 'Iran', 'Thailand', 'Hong'], 'Asia', inplace=True)
X3['native-country'].replace(['Ecuador', 'Columbia', 'Peru','Puerto-Rico', 'Mexico', 'Cuba', 'Jamaica', 'Dominican-Republic', 'Haiti', 'Guatemala', 'Honduras', 'El-Salvador', 'Nicaragua', 'Trinadad&Tobago', 'Panama'], 'Central & South America', inplace=True)

# One-Hot-Encoding for categorical columns
X3 = pd.get_dummies(X3, columns=['marital-status'], dtype=int)
X3 = pd.get_dummies(X3, columns=['workclass'], dtype=int)
X3 = pd.get_dummies(X3, columns=['occupation'], dtype=int)
X3 = pd.get_dummies(X3, columns=['relationship'], dtype=int)
X3 = pd.get_dummies(X3, columns=['race'], dtype=int)
X3 = pd.get_dummies(X3, columns=['native-country'], dtype=int)

# Encode sex 1 or 0
X3['sex'] = np.where(X3['sex'] == 'Male', 1, 0)

print('\n================== features dataframe after preprocessing ==================')
display(X3.head())

print('\n================== features dataframe shape after preprocessing ==================')
print(X3.shape)


################
## Visualizations

int_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
for i in int_columns:
  sns.boxplot(x = df[i])
  plt.show()

  high_income_df = df[df['income>50K'] == 1]

# Group by country and calculate mean age
mean_age_by_country = high_income_df.groupby('native-country')['age'].mean().sort_values()

# Plotting
plt.figure(figsize=(10, 10))
mean_age_by_country.plot(kind='barh')
plt.title('Mean Age with >50K Income by Country')
plt.xlabel('Mean Age')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))

income_count_by_education = high_income_df['education-num'].value_counts().sort_index()
income_count_by_education.plot(kind='bar')
plt.title('>50K Count vs Education Level')
plt.xlabel('Education Level')
plt.ylabel('>50K Income Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting
plt.figure(figsize=(13, 13))
sns.barplot(x="workclass", y="age", hue='income>50K', data=df, palette="viridis")
plt.xlabel("Workclass")
plt.ylabel("Age")
plt.title("Workclass vs Age by Income")
plt.show()







##################
## Classifier Implementation

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X3, np.ravel(y3), test_size=0.2, random_state=1)

##################
## Logistic Regression

print('\n================== Logistic Regression ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Logistic Regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),               # Feature scaling
    ('logreg', LogisticRegression(max_iter=1000))  # Logistic Regression classifier
])

# Define the parameter grid
param_grid = {
    'logreg__C': np.logspace(-4, 4, 20),               # Regularization strength
    'logreg__solver': ['liblinear', 'lbfgs'],  # Solvers
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

# Get the coefficients of the features from the logistic regression model within the pipeline
feature_importances = best_model.named_steps['logreg'].coef_[0]

# Print feature names and their coefficients
print('\n================== feature importance ==================')
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")


##################
## Decision Tree
    
print('\n================== Decision Tree ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Decision Tree model
pipeline = Pipeline([
    ('dtree', DecisionTreeClassifier())  # Decision Tree classifier
])

# Define the parameter grid
param_grid = {
    'dtree__max_depth': np.arange(3, 16),  
    'dtree__min_samples_split': [2, 5, 10, 20],         
    'dtree__criterion': ['gini', 'entropy']             
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)
y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

feature_importances = best_model.named_steps['dtree'].feature_importances_

# Print feature names and their importances
print('\n================== feature importance ==================')

for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")


##################
## Random Forest Classifier

print('\n================== Random Forest Classifier ==================')
print('\n======================================================')

# Create a pipeline that standardizes the data then creates a Random Forest Classifier model
pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

# Define the parameter grid
param_grid = {
    'rf__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'rf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'rf__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('\n================== Best parameters found ==================')
print(grid_search.best_params_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))

# Get the feature importances from the Random Forest model within the pipeline
feature_importances = best_model.named_steps['rf'].feature_importances_

# Print feature names and their importances
print('\n================== Feature Importance ==================')
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")



##################
## Support Vector Machine

# Create a pipeline that standardizes the data then creates an SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('svm', SVC())                 
])

# Define the parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],                # Regularization parameter
    'svm__kernel': ['linear', 'rbf'],     # Type of SVM kernel
    'svm__gamma': ['scale', 'auto']       # Kernel coefficient
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

y_pred = best_model.predict(X_test)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))




# Initialize the XGBClassifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage used to prevent overfitting
    'max_depth': [3, 4, 5],  # Maximum depth of a tree
    'colsample_bytree': [0.7, 0.8],  # Subsample ratio of columns when constructing each tree
    'subsample': [0.7, 0.8]  # Subsample ratio of the training instances
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Evaluate the best model found by GridSearchCV on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print('\n================== test accuracy ==================')
print(test_accuracy)

# Print Confusion Matrix
print('\n================== confusion matrix ==================')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print('\n================== classification report ==================')
print(classification_report(y_test, y_pred))