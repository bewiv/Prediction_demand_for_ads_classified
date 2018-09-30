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

(data['price']==0).value_counts()


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
        print("RMSE for test data: {0:.4f}\n".format(metrics.r2_score(y_test, mdl.predict(X_test))))
        print("RMSE for test data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, mdl.predict(X_test)))))


# ## feature scaling

'''data = data.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
           'item_seq_number','image','image_top_1'], axis=1)
data_sc = data
sc = StandardScaler()
data_sc = sc.fit_transform(data_sc)
X_col = data.columns
data = pd.DataFrame(data_sc, columns=X_col)'''


X_train, X_test = train_test_split(data, test_size=0.2)
y_train = X_train['deal_probability']
y_test = X_test.deal_probability
X_train.drop('deal_probability',1, inplace=True)
X_test.drop('deal_probability',1, inplace=True)

train = X_train
test = X_test

train = train.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)

test = test.drop(['item_id','user_id','city', 'param_1','param_2','param_3','params','title','description','activation_date',
            'item_seq_number','image','image_top_1'], axis=1)


# ## prediction


# RF
rf = RandomForestRegressor(bootstrap = 'True', max_depth = None,max_features = 2, min_samples_leaf = 3,
                           min_samples_split = 10, n_estimators = 200, random_state=42)
rf.fit(train,y_train)


# prediciton train
en_en = pd.DataFrame()
en_en['xgb'] = (xgb_model.predict(dtrain)).tolist()
en_en['lgb'] = (lgb_model.predict(train, num_iteration=lgb_model.best_iteration)).tolist()
en_en['rf'] = (rf.predict(train)).tolist()
en_en['et'] = (et.predict(train)).tolist()
#en_en['dtr'] = (dtr.predict(train)).tolist()


# prediciton test
en_en_test = pd.DataFrame()
en_en_test['xgb'] = (xgb_model.predict(dtest)).tolist()
en_en_test['lgb'] = (lgb_model.predict(test, num_iteration=lgb_model.best_iteration)).tolist()
en_en_test['rf'] = (rf.predict(test)).tolist()
en_en_test['et'] = (et.predict(test)).tolist()
#en_en_test['dtr'] = (dtr.predict(test)).tolist()

en_en.head()

#en_xgb_lgb = en_en[['xgb','rf']]
#en_xgb_lgb_test = en_en_test[['xgb','rf']]

'''from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(en_en[['xgb', 'lgb','rf', 'is_deal_median']], y_train)
'''
dtr = tree.DecisionTreeRegressor(criterion='mse', max_leaf_nodes=7,min_samples_leaf=24,min_samples_split=2,random_state=42)
dtr = dtr.fit(en_en, y_train)

'''dtra = xgb.DMatrix(en_en, label=y_train)
dtes = xgb.DMatrix(en_en_test, label=y_test)
watchlist = [(dtra, 'train'), (dtes, 'test')]
xgb_pars = {'objective' : "reg:logistic",'booster' : "gbtree",'eval_metric' : "rmse",'nthread' : 4,'eta':0.05,
          'max_depth':15,'min_child_weight': 2,'gamma' :0,'subsample':0.7,'colsample_bytree':0.7,'aplha':0,'lambda':0}        
xgb_m = xgb.train(xgb_pars, dtra,2000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=1)
print('Modeling RMSE %.5f' % xgb_m.best_score)'''


get_ipython().run_cell_magic('time', ' 0.23864', 'en_en_test.head()')


print("RMSE for test data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test,dtr.predict(en_en_test)))))


del data#, df1


#blending


concat_sub_train= en_en
concat_sub_test= en_en_test

concat_sub_test['is_deal_max'] = concat_sub_test.iloc[:, 0:3].max(axis=1)
concat_sub_test['is_deal_min'] = concat_sub_test.iloc[:, 0:3].min(axis=1)
concat_sub_test['is_deal_mean'] = concat_sub_test.iloc[:, 0:3].mean(axis=1)
concat_sub_test['is_deal_median'] = concat_sub_test.iloc[:, 0:3].median(axis=1)

concat_sub_train['is_deal_max'] = concat_sub_train.iloc[:, 0:3].max(axis=1)
concat_sub_train['is_deal_min'] = concat_sub_train.iloc[:, 0:3].min(axis=1)
concat_sub_train['is_deal_mean'] = concat_sub_train.iloc[:, 0:3].mean(axis=1)
concat_sub_train['is_deal_median'] = concat_sub_train.iloc[:, 0:3].median(axis=1)


concat_sub_train.head()

print("RMSE for test max data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, d))))
'''print("RMSE for test min data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub_train['is_deal_min']))))
print("RMSE for test mean data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub_train['is_deal_mean']))))
print("RMSE for test median data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub_train['is_deal_median']))))'''

en_en_test.boxplot(column=['xgb', 'lgb','rf', 'et'])

#df2 = pd.DataFrame(np.random.rand(10, 4), column=['xgb', 'lgb','rf', 'et'])
concat_sub_train.head()

cutoff_lo = 0.8
cutoff_hi = 0.2

concat_sub_train['stack_pushout_median'] = np.where(np.all(concat_sub_train.iloc[:,0:3] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub_train.iloc[:,0:3] < cutoff_hi, axis=1),
                                             0, concat_sub_train['is_deal_median']))

concat_sub_train['stack_minmax_mean'] = np.where(np.all(concat_sub_train.iloc[:,0:3] > cutoff_lo, axis=1), 
                                    concat_sub_train['is_deal_max'], 
                                    np.where(np.all(concat_sub_train.iloc[:,0:3] < cutoff_hi, axis=1),
                                             concat_sub_train['is_deal_min'],concat_sub_train['is_deal_mean']))

concat_sub_train['stack_minmax_median'] = np.where(np.all(concat_sub_train.iloc[:,0:3] > cutoff_lo, axis=1), 
                                    concat_sub_train['is_deal_max'], 
                                    np.where(np.all(concat_sub_train.iloc[:,0:3] < cutoff_hi, axis=1),
                                             concat_sub_train['is_deal_min'], concat_sub_train['is_deal_median']))


concat_sub_test['stack_pushout_median'] = np.where(np.all(concat_sub_test.iloc[:,0:3] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub_test.iloc[:,0:3] < cutoff_hi, axis=1),
                                             0, concat_sub_test['is_deal_median']))

concat_sub_test['stack_minmax_mean'] = np.where(np.all(concat_sub_test.iloc[:,0:3] > cutoff_lo, axis=1), 
                                    concat_sub_test['is_deal_max'], 
                                    np.where(np.all(concat_sub_test.iloc[:,0:3] < cutoff_hi, axis=1),
                                             concat_sub_test['is_deal_min'],concat_sub_test['is_deal_mean']))

concat_sub_test['stack_minmax_median'] = np.where(np.all(concat_sub_test.iloc[:,0:3] > cutoff_lo, axis=1), 
                                    concat_sub_test['is_deal_max'], 
                                    np.where(np.all(concat_sub_test.iloc[:,0:3] < cutoff_hi, axis=1),
                                             concat_sub_test['is_deal_min'], concat_sub_test['is_deal_median']))


d1 = en_en_test['xgb']
d2 = en_en_test['lgb']

d1 = (d1+d2)/2
d2 = (d1+d2)/2


d = (d1+d2)/2

print("RMSE for test max data: {0:.4f}\n".format(sqrt(mean_squared_error(y_train, b))))

y_test.shape


# # Stacking ###########################

d1=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_rf2284.csv')
d2=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_xgb2256.csv')
d3=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_lgb2265.csv')

'''d3['deal_probability_1']=d2['SC_xgb']
d3['deal_probability_0']=d3['SC_lgb']
d3['deal_probability_2']=d1['SC_rf']'''
d1['deal_probability_0']=d2['SC_xgb']
d1['deal_probability_1']=d1['SC_rf']


#d3 = d3.drop(['SC_lgb'], axis=1)
d1=d1.drop(['SC_rf','y'], axis=1) # combine 2
d1.head()

dt1=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_rf_test2284.csv')
dt2=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_xgb_test2256.csv')
dt3=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_lgb_test2265.csv')
'''dt3['deal_probability_1']=dt2['SC_xgb']
dt3['deal_probability_0']=dt3['SC_lgb']
dt3['deal_probability_2']=dt1['SC_rf']
dt3 = dt3.drop(['SC_lgb'], axis=1)'''
dt1['deal_probability_0']=dt2['SC_xgb']
dt1['deal_probability_1']=dt1['SC_rf']
dt1=dt1.drop(['SC_rf','y'], axis=1)


dt1.head()


'''############################## 
dtrain = xgb.DMatrix(d1, label=y_train)
dtest = xgb.DMatrix(dt1, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
xgb_pars = {'objective' : "reg:logistic",'booster' : "gbtree",'eval_metric' : "rmse",'nthread' : 4,'eta':0.05,
          'max_depth':15,'min_child_weight': 2,'gamma' :0,'subsample':0.7,'colsample_bytree':0.7,'aplha':0,'lambda':0}        
xgb_model = xgb.train(xgb_pars, dtrain,2000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=1)
print('Modeling RMSE %.5f' % xgb_model.best_score)
''' 
############################### 
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(d1, y_train) 

en_test = pd.DataFrame()
en_test = dt1

en_test.head()

#en_test['combined']= xgb_model.predict(dtest) # Xgboost
en_test['combined'] = clf.predict(dt1) # MLP

print("RMSE for test data: {0:.4f}\n".format(sqrt(mean_squared_error(y, en_test['combined']))))

y.head()

data=[]
y=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/y.csv')
data.insert(0, {'542593': 542593, '0.76786': 0.76786})
y = pd.concat([pd.DataFrame(data), y], ignore_index=True)
y = y.drop(['542593'], axis = 1)
y.columns=['y']


y.head()


# #### split training data to train and validation 

training, valid, y_training, y_valid = train_test_split(train, y_train, test_size=0.2, random_state=42)


lr = LinearRegression(copy_X=True,fit_intercept=True, normalize=True)
lr.fit(training, y_training)

dtr = tree.DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=7, 
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None, min_samples_leaf=24,min_samples_split=2, 
                                 min_weight_fraction_leaf=0.0,presort=False, 
                                 random_state=42, splitter='best')
dtr = dtr.fit(train, y_train)


en_en = pd.DataFrame()


xgb_pred = xgb_model.predict(dtrain)
lgb_pred = model.predict(train, num_iteration=model.best_iteration)


xgb = xgb_pred.tolist()
lgb = lgb_pred.tolist()


en_en['xgb_pred'] = xgb
en_en['lgb_pred'] = lgb
col_name = en_en.columns


en_en = pd.concat([en_en, pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
en_en.head()


# #### Meta Classifier


m_r = tree.DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=7, 
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None, min_samples_leaf=24,min_samples_split=2, 
                                 min_weight_fraction_leaf=0.0,presort=False, 
                                 random_state=42, splitter='best')
m_r = m_r.fit(en_en[['xgb_pred','lgb_pred']], en_en['deal_probability'])


train.head()

en_en_test = pd.DataFrame()

en_en_test['xgb_pred'] = (xgb_model.predict(dtest)).tolist()
en_en_test['lgb_pred'] = (model.predict(test, num_iteration=model.best_iteration)).tolist()
#col_name = en_en.columns

en_en_test['combined'] = m_r.predict(en_en_test[['xgb_pred','lgb_pred']])


en_en_test = pd.concat([en_en_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)


print_score(m_r, en_en[['xgb_pred','lgb_pred']], en_en['deal_probability'], 
            en_en_test[['xgb_pred','lgb_pred']], en_en_test['deal_probability'], train=False)


# ### Single Classifier

en_en


from sklearn.ensemble import BaggingRegressor

bag_clf = BaggingRegressor(base_estimator=lr)

bag_clf.fit(train, y_train.ravel())

print_score(bag_clf, train, y_train, test, y_test, train=False)


del X_train, X_test


# ## blanding

d1=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_rf_test2284.csv')
d2=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_xgb_test2256.csv')
d3=pd.read_csv('E:/PROJET/Avito_Demand_Prediction/pred/sc/SC_lgb_test2265.csv')


d3=d3.drop(['SC_lgb'],axis=1 )


import os
import numpy as np 
import pandas as pd 
from subprocess import check_output


d3['deal_probability_1']=d2['SC_xgb']
d3['deal_probability_0']=d3['SC_lgb']
d3['deal_probability_2']=d1['SC_rf']

d3.head()

d3=d3.drop(['SC_lgb'],axis=1 )


d3.corr()

concat_sub=d3

concat_sub['is_deal_max'] = concat_sub.iloc[:, 0:2].max(axis=1)
concat_sub['is_deal_min'] = concat_sub.iloc[:, 0:2].min(axis=1)
concat_sub['is_deal_mean'] = concat_sub.iloc[:, 0:2].mean(axis=1)
concat_sub['is_deal_median'] = concat_sub.iloc[:, 0:2].median(axis=1)


concat_sub.head()
#concat_sub.drop(['SC_lgb'], axis=1)


print("RMSE for test max data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub['is_deal_max']))))
print("RMSE for test min data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub['is_deal_min']))))
print("RMSE for test mean data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub['is_deal_mean']))))
print("RMSE for test median data: {0:.4f}\n".format(sqrt(mean_squared_error(y_test, concat_sub['is_deal_median']))))

