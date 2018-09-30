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





print("Reading Data......")
data = pd.read_csv('E:/PROJET/Avito_Demand_Prediction/input/train.csv')
print("Reading Done....")


# check the missing value 
def missingvalues(d):
    total = d.isnull().sum().sort_values(ascending = False)
    percent = (d.isnull().sum()*100/d.isnull().count()).sort_values(ascending = False)
    missing_train_data =pd.concat([total, percent], axis = 1, keys=['total', 'percent'])
    return missing_train_data

missingvalues(data).head(10)


def fill_na(d):
    d = d.fillna({
        'price' : d.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
        'price' : d.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
        'price' : d.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median())),
        'param_1' : '',
        'param_2' : '',
        'param_3' : '',
        'description' : ''    
    })
    return d
data = fill_na(data)


data['price'] = data['price'].replace(0, np.nan) 
data = data.fillna({
    'price' : data.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median())),
    'price' : data.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
})


(data['price']==0).value_counts()


# # Feature engineering

def cols_datetime(d):
    d['activation_date'] = pd.to_datetime(d['activation_date'])
    d["month"] = d["activation_date"].dt.month
    d['weekday'] = d['activation_date'].dt.weekday
    d["month_Day"] = d['activation_date'].dt.day
    d["year_Day"] = d['activation_date'].dt.dayofyear
    return d

def exist_img(d):
    d['has_image'] = 1
    d.loc[d['image'].isnull(),'has_image'] = 0
    d['has_image_top_1'] = 1
    d.loc[d['image_top_1'].isnull(),'has_image'] = 0
    return d


def combine_params(d):
    d['params'] = d['param_1'].fillna('') + ' ' + d['param_2'].fillna('') + ' ' + d['param_3'].fillna('')
    d['params'] = d['params'].str.strip()
    return d

data = cols_datetime(data)
data = exist_img(data)
data = combine_params(data)




#count the number of ad that contain zero in deal probability
#((train_df['deal_probability'] == 0).sum()*100)/((train_df['deal_probability'] == 0).count())


# Average Word Length
# We will also extract another feature which will calculate the average word length of each tweet. This can also potentially help us in improving our model.
# 
# Here, we simply take the sum of the length of all the words and divide it by the total length of the tweet:

def avg_word(sentence): # 1.3 Average Word Length
    words = sentence.split()
    if len(words) == 0:
        return 0
    else:
         return (sum(len(word) for word in words)/len(words))

def avg_words(df):
    
    cols = ['title', 'description',  'params', 'param_1', 'param_2', 'param_3']
    for col in cols:
        df['avg_word_' + str(col)] = df[col].apply(lambda x: avg_word(x))

    return df

data = avg_words(data)


from nltk.corpus import stopwords

stop = stopwords.words('russian')

def text_processing(df):
    
    cols = ['title', 'description',  'params','param_1', 'param_2', 'param_3']
    for col in cols:
        df['words_' + str(col)] = df[col].apply(lambda x: len(x.split(" "))) #1.1 Number of Words
        df['len_' + str(col)] = df[col].apply(lambda x: len(x)) #1.2 Number of characters
        df['numerics_' + str(col)] = df[col].apply(lambda x: len([x for x in x.split() if x.isdigit()])) #1.6 Number of numerics
        df['upper_' + str(col)] = df[col].apply(lambda x: len([x for x in x.split() if x.isupper()])) # 1.7 Number of Uppercase words
        df[str(col)] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split())) #2.1 Lower case
        #df[str(col)] = df[col].str.replace('[^\w\s]','') #2.2 Removing Punctuation including ‘#’ and ‘@
        df[str(col)] = df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #2.3 Removal of Stop Words
        #df['stopwords' + str(col)] = df['col'].apply(lambda x: len([x for x in x.split() if x in stop])) 1.4 Number of stopwords
        #df['hastags' + str(col)] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))1.5 Number of special characters   
        df[str(col)] = df[col].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))

    return df
data = text_processing(data)

'''        
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
        '''




data['symbol1_count'] = data['description'].str.count('↓')
data['symbol2_count'] = data['description'].str.count('\*')
data['symbol3_count'] = data['description'].str.count('✔')
data['symbol4_count'] = data['description'].str.count('❀')
data['symbol5_count'] = data['description'].str.count('➚')
data['symbol6_count'] = data['description'].str.count('ஜ')
data['symbol7_count'] = data['description'].str.count('.')
data['symbol8_count'] = data['description'].str.count('!')
data['symbol9_count'] = data['description'].str.count('\?')
data['symbol10_count'] = data['description'].str.count(' ')
data['symbol11_count'] = data['description'].str.count('-')
data['symbol12_count'] = data['description'].str.count(',')



#freq = pd.Series(' '.join(train_df['description']).split()).value_counts()[-1000:]



# agregate feature 


data['user_price_mean'] = data.groupby('user_id')['price'].transform('mean')
data['user_ad_count'] = data.groupby('user_id')['price'].transform('sum')

data['region_price_mean'] = data.groupby('region')['price'].transform('mean')
data['region_price_median'] = data.groupby('region')['price'].transform('median')
data['region_price_max'] = data.groupby('region')['price'].transform('max')

data['region_price_mean'] = data.groupby('region')['price'].transform('mean')
data['region_price_median'] = data.groupby('region')['price'].transform('median')
data['region_price_max'] = data.groupby('region')['price'].transform('max')

data['city_price_mean'] = data.groupby('city')['price'].transform('mean')
data['city_price_median'] = data.groupby('city')['price'].transform('median')
data['city_price_max'] = data.groupby('city')['price'].transform('max')

data['parent_category_name_price_mean'] = data.groupby('parent_category_name')['price'].transform('mean')
data['parent_category_name_price_median'] = data.groupby('parent_category_name')['price'].transform('median')
data['parent_category_name_price_max'] = data.groupby('parent_category_name')['price'].transform('max')

data['category_name_price_mean'] = data.groupby('category_name')['price'].transform('mean')
data['category_name_price_median'] = data.groupby('category_name')['price'].transform('median')
data['category_name_price_max'] = data.groupby('category_name')['price'].transform('max')

data['user_type_category_price_mean'] = data.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
data['user_type_category_price_median'] = data.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
data['user_type_category_price_max'] = data.groupby(['user_type', 'parent_category_name'])['price'].transform('max')



tfidf_vec = TfidfVectorizer(max_features=100,ngram_range=(1,1))
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack') # arpack random harvard


full_tfidf_desc=tfidf_vec.fit_transform(data.description.values.tolist() + data.description.values.tolist())
full_tfidf_title=tfidf_vec.fit_transform(data.title.values.tolist() + data.title.values.tolist())
full_tfidf_params=tfidf_vec.fit_transform(data.params.values.tolist() + data.params.values.tolist())


data_tfidf_desc = tfidf_vec.transform(data.description.values.tolist())
data_tfidf_title = tfidf_vec.transform(data.title.values.tolist())
data_tfidf_params = tfidf_vec.transform(data.params.values.tolist())



svd_obj.fit(full_tfidf_title) # Fit LSA model on full_tfidf data .
data_svd_title = pd.DataFrame(svd_obj.transform(data_tfidf_title)) # Perform dimensionality reduction on train_tfidf.
data_svd_title.columns = ['svd_title_'+str(i+1) for i in range(n_comp)] # create n_comp = 5 feature svd Components for train_svd

data.reset_index(drop=True, inplace=True)
data_svd_title.reset_index(drop=True, inplace=True)

data_df = pd.concat([data, data_svd_title], axis=1) # merge train_df with train_svd
del full_tfidf_title, data_tfidf_title, data_svd_title # delete unsful variable 


svd_obj.fit(full_tfidf_desc) # Fit LSA model on full_tfidf data .
data_svd_desc = pd.DataFrame(svd_obj.transform(data_tfidf_desc)) # Perform dimensionality reduction on train_tfidf.
data_svd_desc.columns = ['svd_description_'+str(i+1) for i in range(n_comp)] # create n_comp = 5 feature svd Components for train_svd

data.reset_index(drop=True, inplace=True)
data_svd_desc.reset_index(drop=True, inplace=True)

data = pd.concat([data, data_svd_desc], axis=1) # merge train_df with train_svd
del full_tfidf_desc, data_tfidf_desc, data_svd_desc # delete unsful variable 


svd_obj.fit(full_tfidf_params) # Fit LSA model on full_tfidf data .
data_svd_params = pd.DataFrame(svd_obj.transform(data_tfidf_params)) # Perform dimensionality reduction on train_tfidf.
data_svd_params.columns = ['svd_params_'+str(i+1) for i in range(n_comp)] # create n_comp = 5 feature svd Components for train_svd

data.reset_index(drop=True, inplace=True)
data_svd_params.reset_index(drop=True, inplace=True)

train_df = pd.concat([data, data_svd_params], axis=1) # merge train_df with train_svd
del full_tfidf_params, data_tfidf_params, data_svd_params # delete unsful variable 


# encode Categorical features
# in this step i'll encode categorical features to numerical


from sklearn.preprocessing import LabelEncoder

cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type","params","param_1","param_2","param_3"]
for col in cat_vars:
    lbl = LabelEncoder()
    lbl.fit(list(data[col].values.astype('str')))
    data[col] = lbl.transform(list(data[col].values.astype('str')))

print("done")

