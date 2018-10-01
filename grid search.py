import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import r2_score

data = pd.read_csv("E:/PROJET/Avito_Demand_Prediction/input/data_prep2000.csv")

data['price'] = data['price'].replace(0, np.nan) 
data = data.fillna({
    'price' : data.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
})



(data['price']==0).value_counts()





data = data.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)


# # prediction


from sklearn.cross_validation import cross_val_score
from sklearn.metrics import explained_variance_score

def print_score(mdl, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("RMSE: {0:.4f}\n".format(sqrt(mean_squared_error(y_train, mdl.predict(X_train)))))

        res = np.sqrt(-cross_val_score(mdl, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        print("Cross V RMSE: \t {0:.4f}".format(np.mean(res)))
        #print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("RMSE: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, mdl.predict(X_test)))))


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test = train_test_split(data, test_size=0.2)
y_train = X_train['deal_probability']
y_test = X_test.deal_probability
X_train.drop('deal_probability',1, inplace=True)
X_test.drop('deal_probability',1, inplace=True)


train = X_train
train = train.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)
test = X_test
test = test.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)



# ## XGBoost

# In[64]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# grid search



import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# ### linear regression 


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

lr = LinearRegression()
parameters = {'fit_intercept':('True', 'False'), 
              'normalize':('True', 'False'), 
              'copy_X':('True', 'False')}

lrgs = GridSearchCV(lr, parameters)
lrgs.fit(train, y_train)

lrgs.best_params_

#X = train
#sc_x = StandardScaler()
#X_std_train = sc_x.fit_transform(X)


from sklearn.linear_model import LinearRegression
# 1. Set up the model
lr = LinearRegression(copy_X=True,fit_intercept=True, normalize=True)
# 2. Use fit
lr.fit(train, y_train)
# 3. Check the score
print("accuracy train:", lr.score(train, y_train))
print("accuracy test:", lr.score(test, y_test))
y_reg_lintr= lr.predict(train)
y_reg_lints= lr.predict(test)
print('train RMSE: ', sqrt(mean_squared_error(y_train, y_reg_lintr)))
print('test RMSE: ', sqrt(mean_squared_error(y_test, y_reg_lints)))

print_score(lr, train, y_train, test, y_test, train=True)
print_score(lr, train, y_train, test, y_test, train=False)

'''import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,10))
sns.heatmap(train_df.corr(), annot=True)
plt.show()'''

#train_df['city'].hist(bins=10)


# ##  Random Forest
# 
# [paper](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)
# 
# * Ensemble of Decision Trees
# 
# * Training via the bagging method (Repeated sampling with replacement)
#   * Bagging: Sample from samples
#   * RF: Sample from predictors. $m=sqrt(p)$ for classification and $m=p/3$ for regression problems.
# 
# * Utilise uncorrelated trees
# 
# Random Forest
# * Sample both observations and features of training data
# 
# Bagging
# * Samples only observations at random
# * Decision Tree select best feature when splitting a node

# In[84]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn import tree

# grid search 
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train, y_train)

grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test, y_test)


rfr.fit(train, y_train)
y_rfr = rfr.predict(train)

mse = metrics.mean_squared_error(y_train,y_rfr)
print ("train RMSE:               ", np.sqrt(mse))


y_rfrts = rfr.predict(test)
mse = metrics.mean_squared_error(y_test,y_rfrts)
print ("test RMSE:               ", np.sqrt(mse))


print_score(lr, train, y_train, test, y_test, train=True)
print_score(lr, train, y_train, test, y_test, train=False)


print ("feature_importances:",rf_clf.feature_importances_)
print ("n_features:         ",rf_clf.n_features_)
print ("n_outputs:          ",rf_clf.n_outputs_)


# ## Decision Tree

from sklearn import tree
tree_reg = tree.DecisionTreeRegressor(random_state=42)
tree_reg = tree_reg.fit(train, y_train)

y_1 = tree_reg.predict(train)
y_2 = tree_reg.predict(test)

mse = metrics.mean_squared_error(y_train,y_1)
mse1 = metrics.mean_squared_error(y_test,y_2)

print_score(tree_reg, train, y_train, test, y_test, train=True)
print_score(tree_reg, train, y_train, test, y_test, train=False)

#grid search
from sklearn.model_selection import GridSearchCV


# In[105]:


params = {'max_leaf_nodes': list(range(2, 100)),
          'min_samples_split': [2, 5, 10, 20],
          'max_depth': [None, 2, 5, 10, 15, 20],
          'min_samples_leaf': list(range(5, 50))}


grid_search_cv = GridSearchCV(tree.DecisionTreeRegressor(random_state=42), params, n_jobs=-1, verbose=1)

grid_search_cv.fit(train, y_train)

grid_search_cv.best_estimator_


###########################################
y_pred_dt = grid_search_cv.predict(test)



from sklearn import tree
tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=4, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=12,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=42, splitter='best')
tree_reg = tree_reg.fit(train, y_train)




print_score(tree_reg, train, y_train, test, y_test, train=True)
print_score(tree_reg, train, y_train, test, y_test, train=False)


# ## SVM 



from sklearn import svm
svr_m = svm.SVR()
svr_m.fit(train,y_train) 



print_score(svr_m, train, y_train, test, y_test, train=True)
print_score(svr_m, train, y_train, test, y_test, train=False)


svr_m.get_params()


# ##### Grid Search


from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler


X = data
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X)
X_col = data.columns
df1 = pd.DataFrame(X, columns=X_col)
X_train, X_test = train_test_split(df1, test_size=0.2)
y_train = X_train['deal_probability']
y_test = X_test.deal_probability
X_train.drop('deal_probability',1, inplace=True)
X_test.drop('deal_probability',1, inplace=True)


from sklearn import svm
parameters = {'kernel': ('linear', 'rbf','poly'), 
              'C':[1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
              'epsilon':[0.1,0.2,0.5,0.3]}




svr = svm.SVR()
clf = GridSearchCV(svr, parameters, cv=3)
clf.fit(train,y_train)


clf.best_params_


# ### Adaboost


from sklearn.ensemble import AdaBoostRegressor


ada_clf = AdaBoostRegressor()


ada_clf.fit(train, y_train)



print_score(ada_clf, train, y_train, test, y_test, train=True)


print_score(ada_clf, train, y_train, test, y_test, train=False)


# ### Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor


gbc_clf = GradientBoostingRegressor()
gbc_clf.fit(train, y_train)

print_score(gbc_clf, train, y_train, test, y_test, train=False)


# ### light gbm

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(train)
x_test = sc.transform(test)

import lightgbm as lgb
#custom function to build the LightGBM model.
lgtrain = lgb.Dataset(x_train, label=y_train)
lgval = lgb.Dataset(x_test, label=y_test)
params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 200,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        #'max_depth':-1,
        "min_child_samples":20
       # ,"boosting":"rf"
    }
    
    
evals_result = {}
model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=1500, 
                  verbose_eval=20, evals_result=evals_result)
pred_test_y = model.predict(test, num_iteration=model.best_iteration)



# # NN

from sklearn.neural_network import MLPRegressor

mlpr = MLPRegressor()
mlpr.fit(train,y_train)

print_score(mlpr, train, y_train, test, y_test, train=True)
print_score(mlpr, train, y_train, test, y_test, train=False)