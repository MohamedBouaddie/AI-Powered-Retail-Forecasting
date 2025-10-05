# ## File Descriptions and Data Field Information
# ##### train.csv
# - The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.
# - store_nbr identifies the store at which the products are sold.
# - family identifies the type of product sold.
# - sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
# - onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.
# ##### test.csv
# - The test data, having the same features as the training data. You will predict the target sales for the dates in this file.
# - The dates in the test data are for the 15 days after the last date in the training data.
# ##### sample_submission.csv
# - A sample submission file in the correct format.
# ##### stores.csv
# - Store metadata, including city, state, type, and cluster.
# - cluster is a grouping of similar stores.
# ##### oil.csv
# - Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
# ##### holidays_events.csv
# - Holidays and Events, with metadata
# - NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
# - Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
# ##### Additional Notes
# - Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
# - A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.


import numpy as np
import pandas as pd
import seaborn as sns # type: ignore 

holidays_events = pd.read_csv('holidays_events.csv')
oil = pd.read_csv('oil.csv')
stores = pd.read_csv('stores.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
transactions = pd.read_csv('transactions.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# ##### List out feature names of each dataset
holidays_events.columns
holidays_events.head()

oil.head()
oil.columns

stores.head()
stores.columns

train.head()
train.columns

test.head()
test.columns

transactions.head()
transactions.columns

train.info()
train.describe()
train.isnull().sum()


print('Diamensions of train data : {}'.format(train.shape))
print('Diamensions of test data : {}'.format(test.shape))

train.columns
train.head()
train.corr()


# ###### The dependant data to be predicted is the sales feature.
train['sales'].unique()
train['sales'].value_counts()

#### Feature Engineering
train1 = train.merge(oil, on = 'date', how='left')
train1 = train1.merge(holidays_events, on = 'date', how='left')
train1 = train1.merge(stores, on = 'store_nbr', how='left')
train1 = train1.merge(transactions, on = ['date', 'store_nbr'], how='left')
train1 = train1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})
test1 = test.merge(oil, on = 'date', how='left')
test1 = test1.merge(holidays_events, on = 'date', how='left')
test1 = test1.merge(stores, on = 'store_nbr', how='left')
test1 = test1.merge(transactions, on = ['date', 'store_nbr'], how='left')
test1 = test1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})

train1.head()
test1.head()

print("train predictions :",train1.columns)
print("test predictions :",test1.columns)
train1["family"].value_counts()
train1["city"].value_counts()
train1["state"].value_counts()
train1["onpromotion"].value_counts()
train1["store_type"].value_counts()


corr = train1.corr()
sns.heatmap(corr)

sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'store_nbr',y = 'sales',data = train1,palette = "Blues")
sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'store_nbr',y = 'transactions',data = train1,palette = "Blues")
sns.set(rc={'figure.figsize':(20,8.27)})
sns.lineplot(x = "transactions",y = 'sales',data = train1,palette = "Blues")
sns.set(rc={'figure.figsize':(20,8.27)})
sns.lineplot(x = "onpromotion",y = 'sales',data = train1,palette = "Blues")
sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'cluster',y = 'transactions',data = train1,palette = "Blues")
from sklearn.model_selection import train_test_split
features=['date','store_nbr','family','onpromotion','dcoilwtico','holiday_type','locale','locale_name','description','transferred','city','state','store_type','cluster','transactions']
X=train1[features]
y=train1.sales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)

# define the model
model = LinearRegression()

# fit the model
model.fit(X, y)

# get importance
importance = model.coef_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

def feature_eng(data):
    data['date'] = pd.to_datetime(data['date'])
    data['dayofweek'] = data['date'].dt.dayofweek
    data['quarter'] = data['date'].dt.quarter
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['dayofyear'] = data['date'].dt.dayofyear
    data['dayofmonth'] = data['date'].dt.day
    return data

train1 = feature_eng(train1)
test1 = feature_eng(test1)
train1.head()
