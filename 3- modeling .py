import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from math import sqrt
from sklearn.metrics import mean_squared_error

from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV 

from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor


data = pd.read_csv("E:/PROJET/Avito_Demand_Prediction/input/data_prep1.csv")


#(data['price']==0).value_counts()


'''data['price'] = data['price'].replace(0, np.nan) 
data = data.fillna({
    'price' : data.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
})'''



'''from sklearn.preprocessing import LabelEncoder

cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type","params","param_1","param_2","param_3"]
for col in cat_vars:
    lbl = LabelEncoder()
    lbl.fit(list(data[col].values.astype('str')))
    data[col] = lbl.transform(list(data[col].values.astype('str')))

print("done")'''


# # prediction

def print_score(mdl, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score rmse
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("RMSE for train data: {0:.4f}\n".format(sqrt(mean_squared_error(y_train, mdl.predict(X_train)))))

        res = np.sqrt(-cross_val_score(mdl, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        print("Cross Validation score RMSE: \t {0:.4f}".format(np.mean(res)))
        #print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("RMSE for test data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, mdl.predict(X_test)))))


# ## feature scaling

'''data = data.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
           'item_seq_number','image','image_top_1'], axis=1)
data_sc = data
sc = StandardScaler()
data_sc = sc.fit_transform(data_sc)
X_col = data.columns
df1 = pd.DataFrame(data_sc, columns=X_col)'''

#df1.head()


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#a = pd.read_csv("E:/PROJET/Avito_Demand_Prediction/input/data_prep2000.csv")
#X_train, X_test = train_test_split(data, test_size=0.2)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
y_train = X_train['deal_probability']
y_test = X_test.deal_probability
X_train.drop('deal_probability',1, inplace=True)
X_test.drop('deal_probability',1, inplace=True)


del data#, df1

train = X_train
train = train.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)
test = X_test
test = test.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)


# ## 1- Linear Regession
# best parameters {'copy_X': 'True', 
#                  'fit_intercept': 'True', 
#                  'normalize': 'True'}

# 1. Set up the model
lr = LinearRegression(copy_X=True,fit_intercept=True, normalize=True)
# 2. Use fit
lr.fit(train, y_train)
# 3. Check the score
#print_score(lr, train, y_train, test, y_test, train=True)
print_score(lr, train, y_train, test, y_test, train=False)


del lr 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(train)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))


# ## 2- Decision Tree
# 
# grid search: max_leaf_nodes=4,random_state=42,min_samples_split=2, min_samples_leaf=24, criterion='mse
# 


dtr = tree.DecisionTreeRegressor(criterion='mse', max_leaf_nodes=7,min_samples_leaf=24,min_samples_split=2,random_state=42)
dtr = dtr.fit(train, y_train)
print_score(dtr, train, y_train, test, y_test, train=False)


#print_score(dtr, train, y_train, test, y_test, train=True)
#print_score(dtr, train, y_train, test, y_test, train=False)



# prediciton 
pred = pd.DataFrame()
pred_test = pd.DataFrame()


'''pred['lr'] = (lr.predict(train)).tolist()
pred_test['lr'] = (lr.predict(test)).tolist()

pred['dt'] = (dtr.predict(train)).tolist()
pred_test['dt'] = (dtr.predict(test)).tolist()

pred['xgb'] = (xgb_model.predict(dtrain)).tolist()
pred_test['xgb'] = (xgb_model.predict(dtest)).tolist()

pred['lgb'] = (lgb_model.predict(train, num_iteration=model.best_iteration)).tolist()
pred_test['lgb'] = (lgb_model.predict(test, num_iteration=model.best_iteration)).tolist()

pred['dt'] = (dtr.predict(train)).tolist()
pred_test['dt'] = (dtr.predict(test)).tolist()'''


pred['SC_lr'] = (bag_clf.predict(train)).tolist()
pred_test['SC_lr'] = (bag_clf.predict(test)).tolist()

pred['y'] = y_train
pred_test['y'] = y_test


pred.to_csv("E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_lr2478.csv", index=False)
pred_test.to_csv("E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_lr_test2478.csv", index=False)


# ## 3- SVM
# C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False

svr_m = svm.SVR(kernel='rbf', epsilon=0.1)

svr_m.fit(train,y_train)

#print_score(svr_m, train, y_train, test, y_test, train=True)
print_score(svr_m, train, y_train, test, y_test, train=False)

# prediciton 
#svr_m_pred = dtr.predict(train)


# ## 4- Random Forest
# grid search: 'bootstrap': 'True','max_depth': None,'max_features': 2,'min_samples_leaf': 3,'min_samples_split': 10,
#  'n_estimators': 200

# Create a based model
rf = RandomForestRegressor(bootstrap = 'True', max_depth = None,max_features = 2, min_samples_leaf = 3,
                           min_samples_split = 10, n_estimators = 200, random_state=42)
rf.fit(train,y_train)

#print_score(rf, train, y_train, test, y_test, train=True)
print_score(rf, train, y_train, test, y_test, train=False)

# prediciton 
rf_pred = rf.predict(train)

# ## Extra tree

from sklearn.ensemble import ExtraTreesRegressor
et = ExtraTreesRegressor(bootstrap = True, max_features = 2,min_samples_leaf = 3, min_samples_split = 10,n_estimators = 200)
et.fit(train, y_train)

print_score(et, train, y_train, test, y_test, train=False)


# ## 5- XGBoot
# booster = 'gbtree', colsample_bytree = 0.8, gamma = 0.5, max_depth = 5, min_child_weight = 1,
#                      nthread = 2, subsample = 1.0

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xs_train = sc.fit_transform(train)
xs_test = sc.transform(test)'''

dtrain = xgb.DMatrix(train, label=y_train)
dtest = xgb.DMatrix(test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
xgb_pars = {'objective' : "reg:logistic",
          'booster' : "gbtree",
          'eval_metric' : "rmse",
          'nthread' : 4,
          'eta':0.05,
          'max_depth':15,
          'min_child_weight': 2,
          'gamma' :0,
          'subsample':0.7,
          'colsample_bytree':0.7,
          'aplha':0,
          'lambda':0}        
xgb_model = xgb.train(xgb_pars, dtrain,2000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=1)
print('Modeling RMSE %.5f' % xgb_model.best_score)

'''
0.22741
x = xgb.XGBRegressor(booster = 'gbtree', colsample_bytree = 0.8, gamma = 0.5, max_depth = 5, min_child_weight = 1,
                     nthread = 2, subsample = 1.0)
x.fit(train, y_train)'''

#print_score(x, train, y_train, test, y_test, train=False)

xgb_pred = xgb_model.predict(dtrain)


# ## Light Gboost

import lightgbm as lgb
#custom function to build the LightGBM model.
lgtrain = lgb.Dataset(xs_train, label=y_train)
lgval = lgb.Dataset(xs_test, label=y_test)
params = {
        "objective" : "regression_l2",
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
lgb_model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=2000, 
                  verbose_eval=20, evals_result=evals_result)
#pred_test_y = lgb_model.predict(test, num_iteration=model.best_iteration)

del xs_test, xs_train 


train.shape


from sklearn.linear_model import Ridge

r = Ridge()
r.fit(train, y_train) 


print_score(r, train, y_train, test, y_test, train=False)


from sklearn.neural_network import MLPRegressor

clf = MLPRegressor()
clf.fit(train, y_train)       


print_score(r, train, y_train, test, y_test, train=False)