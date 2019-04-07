"""Second example is a Titanic Dataset Gradient Boosting and Random Forest classification"""
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from app import two_model_feature_importance_app

# We read a titanic dataset, and encode sex into numbers, as it might be important
df = pd.read_csv('train.csv')
sex_encoder = pd.get_dummies(df['Sex'])
df['IsMale'] = sex_encoder['male']

#Then train - test division
train_indx = df.sample(frac=0.8).index
train_df = df[df.index.isin(train_indx)]
test_df = df[~ df.index.isin(train_indx)]

# for a simplisity, we analize numeric columns only, so other should be omited
columns_to_drop = ['Survived', 'Name', 'PassengerId', 'Cabin', 'Sex', 'Ticket', 'Embarked']
X_train = train_df.drop(columns_to_drop, axis = 1).values
y_train = train_df['Survived'].values
X_test = test_df.drop(columns_to_drop, axis = 1).values
y_test = test_df['Survived'].values

# Time to input something into null values, then build, name, and train the models
model1 = make_pipeline(
    SimpleImputer(strategy='mean'),
    RandomForestClassifier(n_estimators = 100))
model2 = make_pipeline(
    SimpleImputer(strategy='mean'),
    GradientBoostingClassifier())

model1.name = 'Random Forest'
model2.name = 'XGBoost'

for model in (model1, model2):
    model.fit(X_train, y_train)
    # The Pipeline model has not feature_importances_ attribute, so we need to define it manually:
    model.feature_importances_ = model.steps[1][1].feature_importances_

#Now we are ready to run the app. We change the port so we can see the Iris and Titanic app's on the same time.
two_model_feature_importance_app(model1, model2, list(train_df.drop(columns_to_drop, axis = 1).columns), port = 8040)