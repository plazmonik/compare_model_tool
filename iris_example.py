# At first, let's try to use the application for a famous Iris dataset;

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from app import two_model_feature_importance_app


# The data is available in sklearn.dataset
dict_ = datasets.load_iris()

# We split data into train and test set, as every good Data Scientist would do:
X_train, X_test, y_train, y_test = train_test_split(dict_['data'], dict_['target'])


# Then we define and train the models:
model1 = RandomForestClassifier(n_estimators = 100)
model1.name = 'Random Forest'
model2 = DecisionTreeClassifier()
model2.name = 'Decision Tree'

for model in (model1, model2):
    model.fit(X_train, y_train)

two_model_feature_importance_app(model1, model2, dict_['feature_names'])