#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:41:00 2018

@author: miaowang
"""

import os
import numpy as np
import pandas as pd
import datetime

import sklearn
from geopy.distance import vincenty
from sklearn.decomposition import SparsePCA, FastICA, FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

"""
Mentioned parameter tuning, but no code for it.
Used a validation set instead of cross validation (which is probably fine for this size of data).
Tried a couple other tree type models, but didn’t do any parameter tuning there.

"""



os.chdir('/Users/miaowang/Box Sync/2017DS/Lyft')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.isnull().sum()
test.isnull().sum()

##### data summary
train.describe()
train.head()

##### create new features based on existing feature
# 1. create a new variable converting Timestamp To Date and Time
train['date'] = [datetime.datetime.fromtimestamp(d).isoformat() for d in train.start_timestamp]
train['pickup_datetime'] = pd.to_datetime(train.date)
test['date'] = [datetime.datetime.fromtimestamp(d).isoformat() for d in test.start_timestamp]
test['pickup_datetime'] = pd.to_datetime(test.date)

# 2. create day, year, week, hour, minute, holiday features
train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
train.loc[:, 'pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear
train.loc[:, 'pickup_weekofyear'] = train['pickup_datetime'].dt.weekofyear
train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
test.loc[:, 'pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear
test.loc[:, 'pickup_weekofyear'] = test['pickup_datetime'].dt.weekofyear
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

# Holidays
holidays = [1, 18, 43, 46, 129, 151, 171, 186, 249, 284, 316, 329, 361]

train['isHoliday'] = [1 if x == 6 else 0 for x in train['pickup_weekday']]
test['isHoliday'] = [1 if x == 6 else 0 for x in test['pickup_weekday']]

for x in holidays:
    train.loc[train.pickup_dayofyear == x,'isHoliday'] = 1
    test.loc[test.pickup_dayofyear == x,'isHoliday'] = 1


# night/day time
train['night_trip'] = [True if x < 7 else False for x in train['pickup_hour']]
train['rush_hour'] = [True if 9 < x < 20 else False for x in train['pickup_hour']]
train['weekday'] = [True if x < 5 else False for x in train['pickup_weekday']]
test['night_trip'] = [True if x < 7 else False for x in test['pickup_hour']]
test['rush_hour'] = [True if 9 < x < 20 else False for x in test['pickup_hour']]
test['weekday'] = [True if x < 5 else False for x in test['pickup_weekday']]

### log Y variable
train['log_trip_duration'] = np.log(train['duration'].values + 1)



# 3. calculate distance as new features
    
def haversine(lat1, lon1, lat2, lon2):
    AVG_EARTH_RADIUS = 6371

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    d = np.sin(.5*(lat2 - lat1))**2 + np.cos(lat1)*np.cos(lat2) * np.sin(.5*(lon2 - lon1))**2
    h = 2*AVG_EARTH_RADIUS*np.arcsin(np.sqrt(d))

    return h

def bearing(lat1, lon1, lat2, lon2):
    AVG_EARTH_RADIUS = 6371

    lon_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    y = np.sin(lon_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_delta_rad)

    return np.degrees(np.arctan2(y, x))

def center_coord(lat1, lon1, lat2, lon2):

    clat = (lat1 + lat2)/2
    clon = (lon1 + lon2)/2

    return clat, clon    


# Distance features
train.loc[:, 'trip_distance'] = haversine(train['start_lat'].values,
                                          train['start_lng'].values,
                                          train['end_lat'].values,
                                          train['end_lng'].values)
train.loc[:, 'trip_direction'] = bearing(train['start_lat'].values,
                                          train['start_lng'].values,
                                          train['end_lat'].values,
                                          train['end_lng'].values)
train.loc[:, 'center_latitude'], train.loc[:, 'center_longitude'] = \
                              center_coord(train['start_lat'].values,
                              train['start_lng'].values,
                              train['end_lat'].values,
                              train['end_lng'].values)

test.loc[:, 'trip_distance'] = haversine(test['start_lat'].values,
                                          test['start_lng'].values,
                                          test['end_lat'].values,
                                          test['end_lng'].values)
test.loc[:, 'trip_direction'] = bearing(test['start_lat'].values,
                                          test['start_lng'].values,
                                          test['end_lat'].values,
                                          test['end_lng'].values)
test.loc[:, 'center_latitude'], test.loc[:, 'center_longitude'] = \
                              center_coord(test['start_lat'].values,
                              test['start_lng'].values,
                              test['end_lat'].values,
                              test['end_lng'].values)

# trip speed: distance/duration
train['speed'] = train['trip_distance']/train['duration']
train['pace'] = train['duration']/train['trip_distance']
                           

###############################################
# visualize the location of pick up 

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (6, 4)
%matplotlib inline

color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')   


fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121)
ax1.scatter(train.start_lng,
            train.start_lat,
            s=1,alpha=0.1,color='red')
ax1.scatter(test.start_lng,
            test.start_lat,
            s=1,alpha=0.1,color='blue')

plt.ylim([40.50,41.00])
plt.xlim([-74.40,-73.60])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Pickup Location (Train vs. Test)',fontsize=18)
plt.legend()


ax1 = fig.add_subplot(122)
ax1.scatter(train.end_lng,
            train.end_lat,
            s=1,alpha=0.1,color='red')
ax1.scatter(test.end_lng,
            test.end_lat,
            s=1,alpha=0.1,color='blue')

plt.ylim([40.50,41.00])
plt.xlim([-74.40,-73.60])
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.title('Dropoff Location (Train vs. Test)',fontsize=18)
plt.legend()

# correlation plot

corr_df = train[['start_lng', 'start_lat', 'end_lng', 'end_lat',
                 'pickup_datetime','pickup_weekday', 'pickup_dayofyear', 
       'pickup_hour', 'pickup_minute', 'isHoliday', 'night_trip', 'rush_hour',
       'log_trip_duration', 'trip_distance', 'trip_direction',
       'center_latitude', 'center_longitude', 'pace']]

corr = corr_df.corr()
plt.figure(figsize=(15,15))
swarm_plot = sns.heatmap(corr,vmax=1,square=True,annot=True)
fig = swarm_plot.get_figure()
fig.savefig('/Users/miaowang/Box Sync/conference/lyft_corr.png') 

# visualize the relationship between duration and distance

sns.jointplot('trip_distance','duration',data=train.ix[:10000,],s=10,alpha=0.5,color='green')

sns.jointplot(np.log10(train["trip_distance"][:10000]+1),np.log10(train["duration"][:10000]+1),s=10,alpha=0.5,color='green')
plt.xlabel('log (distance)')
plt.ylabel('log (trip duration)')


sns.jointplot('trip_distance','pace',data=train.ix[:10000,],s=10,alpha=0.5,color='green')
plot = train[((train.pace < 3.918353e+05) & (train.pace >9.283614e-05))]
sns.jointplot(np.log10(plot["trip_distance"][:10000]+1),np.log10(plot["pace"][:10000]+1),s=10,alpha=0.5,color='green')
plt.xlabel('log (distance)')
plt.ylabel('log (trip duration)')



###############################################
## plot some summary statistics/characteristics                             
colla1 = train.groupby(['isHoliday','pickup_weekday','pickup_dayofyear', 'night_trip', 
                        'rush_hour','pickup_hour']).agg(['median', 'mean', 'count']).reset_index()
colla1.to_csv('colla1.csv')
colla1 = pd.read_csv('colla1.csv')


ax = sns.lmplot(x="trip_distance_median", y="trip_distance_count", hue='rush_hour_', data=colla1, lowess=True)
ax.set(xlabel='Trip Distance (Median)', ylabel='Demand (Total Ride Counts)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_demand_distance.png')


ax = sns.lmplot(x="duration_median", y="trip_distance_count", hue='rush_hour_', data=colla1, lowess=True)
ax.set(xlabel='Trip Duration (Median)', ylabel='Demand (Total Ride Counts)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_demand_duration.png')


ax = sns.lmplot(x="trip_direction_median", y="trip_distance_count", hue='rush_hour_', data=colla1, lowess=True)
ax.set(xlabel='Trip Direction', ylabel='Demand (Total Ride Counts)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_demand_direction.png')

# 132 degrees counter-clockwise from North and therefore negative

ax = sns.violinplot(x="pickup_weekday_", y="log_trip_duration_count", hue="rush_hour_", data=colla1)
ax.set(xlabel='Week Day', ylabel='Demand (Total Ride Counts)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_rush.png')

ax = sns.boxplot(x="pickup_hour_", y="log_trip_duration_count", data=colla1)
ax.set(xlabel='Pick Up Hour', ylabel='Demand (Total Ride Counts)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_hour1.png')

ax = sns.boxplot(x="pickup_hour_", y="trip_distance_median", data=colla1)
ax.set(xlabel='Pick Up Hour', ylabel='Distance (Median)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_hour2.png')

ax = sns.boxplot(x="pickup_hour_", y="duration_median", data=colla1)
ax.set(xlabel='Pick Up Hour', ylabel='Duration (Median)')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_hour3.png')

##################################################

###################################################
### select features in the final model

DO_NOT_USE_FOR_TRAINING = ['row_id', 'start_lng', 'start_lat', 'end_lng', 'end_lat',
       'start_timestamp', 'date', 'pickup_datetime', 'log_trip_duration', 'duration','speed', 'pace']

new_df = train.drop([col for col in DO_NOT_USE_FOR_TRAINING if col in train], axis=1)
new_df_test = test.drop([col for col in DO_NOT_USE_FOR_TRAINING if col in test], axis=1)
new_df.shape, new_df_test.shape
new_df.columns == new_df_test.columns


y = np.log(train['duration'].values)
train_attr = np.array(new_df)
train_attr.shape

"""
only use 1000 data set for tunning purposes
train_attr  = train_attr[:1000,:]
y = y[:1000]
new_df_test = np.array(new_df_test)
new_df_test  = new_df_test[:1000,:]
"""

# visualiza the new features
plt.hist(train['duration'].values, bins=100)
plt.xlabel('(trip_duration)')
plt.ylabel('number of train records')
plt.savefig('/Users/miaowang/Box Sync/third paper/proposal/lyft_f1.png')

N = 10000000
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].plot(train['start_lng'].values[:N], train['start_lat'].values[:N], 'b.',
           label='train', alpha=0.1)
ax[1].plot(test['start_lng'].values[:N], test['start_lat'].values[:N], 'g.',
           label='test', alpha=0.1)
fig.suptitle('Train and Test Pickup Location Visualization')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim([40.5, 41])
plt.xlim([-74.5, -73.5])
plt.show()
#########################################################
# split train data set into train and validation data set
RANDOM_STATE = 42
train_x, val_x, train_y, val_y = train_test_split(train_attr, y_tst, test_size=0.2, random_state = RANDOM_STATE)

# Save some memory, if you have >=6G, just comment this out
del train, train_attr

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(val_x, label=val_y)
dtest = xgb.DMatrix(new_df_test.values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Tune these params, see https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
##########################################################
##### XGB with CV and tune parameters
"""
xgb_pars = {'min_child_weight': 100, 'eta': 0.1, 'colsample_bytree': 0.7, 'max_depth': 15,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'mae', 'objective': 'reg:linear'}
"""
tune_train = train_attr[:10000,]
tune_y = y[:10000]
dtrain = xgb.DMatrix(tune_train, label=tune_y)
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'lambda': 1.,
    'booster' : 'gbtree', 
    'silent': 1,
    # Other parameters
    'objective':'reg:linear',
}

params['eval_metric'] = "mae"
num_boost_round = 999

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=100
)

cv_results['test-mae-mean'].min()

########################################## test purpose, run xgb on subset
#########################################################
# split train data set into train and validation data set
train_attr_tst = train_attr[:1000,]
y_tst = y[:1000]
RANDOM_STATE = 42
train_x, val_x, train_y, val_y = train_test_split(train_attr_tst, y, test_size=0.2, random_state = RANDOM_STATE)

# Save some memory, if you have >=6G, just comment this out
del train, train_attr_tst

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(val_x, label=val_y)
dtest = xgb.DMatrix(new_df_test.values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Tune these params, see https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
##########################################################
##### XGB with CV and tune parameters
"""
xgb_pars = {'min_child_weight': 100, 'eta': 0.1, 'colsample_bytree': 0.7, 'max_depth': 15,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'mae', 'objective': 'reg:linear'}
"""
tune_train = train_attr[:10000,]
tune_y = y[:10000]
dtrain = xgb.DMatrix(tune_train, label=tune_y)
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'lambda': 1.,
    'booster' : 'gbtree', 
    'silent': 1,
    # Other parameters
    'objective':'reg:linear',
}

params['eval_metric'] = "mae"
num_boost_round = 999

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=100
)

cv_results['test-mae-mean'].min()

#############################################3 test purpose, run xgb on subset







##################################### begin tunning
## Parameters max_depth and min_child_weight
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in np.arange(5,20,5)
    for min_child_weight in np.arange(50,150,50)
]

# Define initial best params and MAE
min_mae = float("Inf")
best_params = None
max_depth_xgb=[]
min_child_weight_xgb=[]
mean_mae_xgb = []
boost_rounds_xgb = []

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
    max_depth_xgb.append(max_depth)
    min_child_weight_xgb.append(min_child_weight)
    mean_mae_xgb.append(mean_mae)
    boost_rounds_xgb.append(boost_rounds)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


xgb_tune1 = pd.DataFrame(np.column_stack([max_depth_xgb,
                                                  min_child_weight_xgb, mean_mae_xgb,
                                                  boost_rounds_xgb]), 
                               columns=['max depth', 'min child weight ', 'mae', 'boost rounds'])

cv_results
params['max_depth'] = 15
params['min_child_weight'] = 100

# retrieve performance metrics

epochs = len(cv_results['test-mae-mean'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, cv_results['train-mae-mean'], label='Train')
ax.plot(x_axis, cv_results['test-mae-mean'], label='Test')
ax.legend()
plt.ylim(0,1)
plt.xlabel('# iterations')
plt.ylabel('MAE')
plt.title('XGBoost MAE')
plt.legend(['Training Set','CV set'],loc='upper right')
plt.show()




## Parameters subsample and colsample_bytree

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in np.arange(1,11)]
    for colsample in [i/10. for i in np.arange(1,11)]
]

min_mae = float("Inf")
best_params = None
subsample_xgb=[]
colsample_xgb=[]
mean_mae_xgb = []
boost_rounds_xgb = []
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
    subsample_xgb.append(subsample)
    colsample_xgb.append(colsample)
    mean_mae_xgb.append(mean_mae)
    boost_rounds_xgb.append(boost_rounds)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

xgb_tune2 = pd.DataFrame(np.column_stack([subsample_xgb,
                                                  colsample_xgb, mean_mae_xgb,
                                                  boost_rounds_xgb]), 
                               columns=['subsample', 'colsample', 'mae', 'boost rounds'])



# update our params dictionary
params['subsample'] = .8
params['colsample_bytree'] = .7


# retrieve performance metrics

epochs = len(cv_results['test-mae-mean'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, cv_results['train-mae-mean'], label='Train')
ax.plot(x_axis, cv_results['test-mae-mean'], label='Test')
ax.legend()
plt.ylim(0,1)
plt.xlabel('# training examples')
plt.ylabel('MAE')
plt.title('XGBoost MAE')
plt.legend(['Training Set','CV set'],loc='upper right')
plt.show()





## Parameter ETA

%time
# This can take some time…
min_mae = float("Inf")
best_params = None

for eta in [.2, .1]:
    print("CV with eta={}".format(eta))

    # We update our parameters
    params['eta'] = eta

    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
            )

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta

print("Best params: {}, MAE: {}".format(best_params, min_mae))

params['eta'] = .1

##########################
## finalized model
xgb_pars = {'min_child_weight': 100, 'eta': 0.1, 'colsample_bytree': 0.7, 'max_depth': 15,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'mae', 'objective': 'reg:linear'}

model_xgb = xgb.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=20)

print('XGB Modeling MAE %.5f' % model_xgb.best_score)



xgb.plot_importance(model_xgb, ax=None, height=0.2, xlim=None,
                    ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features',
                    importance_type='weight', max_num_features=None,
                    grid=True)
xgb.plot_importance(model_xgb)
plt.show()

def mae(y_true, y_pred):
    return np.mean(np.absolute(y_pred - y_true), axis=-1)

"""
#Choose all predictors except target & IDcols
#predictors = [x for x in train.columns if x not in [target, IDcol]]

## baseline model
gbm0 = GradientBoostingRegressor(random_state=10)
modelfit(gbm0, train_attr)


## Fix learning rate and number of estimators for tuning tree-based parameters

param_test1 = {'n_estimators':[1,2,3,4,5]}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_attr, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

## Tuning tree-specific parameters

param_test2 = {'max_depth':[5,6,7,8], 'min_samples_split':[100,200,300,400]}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train_attr, y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test3 = {'min_samples_split':[100,200,300], 'min_samples_leaf':[30,40,50]}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train_attr, y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

modelfit(gsearch3.best_estimator_, train_attr)

param_test4 = {'max_features':[7,8,9,10]}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train_attr, y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

## Tuning subsample and making models with lower learning rate

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
param_grid = param_test5, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train_attr, y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

## reduce learning rate

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.05, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
param_grid = param_test5, scoring='neg_median_absolute_error',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train_attr, y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


gbm_tuned_2 = GradientBoostingRegressor(learning_rate=0.05, n_estimators=600,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_2, train_attr)
"""


############### TREE
"""
TREE_REGRESSORS = [
    DecisionTreeRegressor(),
    RandomForestRegressor()
]
models = []
for regressor in TREE_REGRESSORS:
    clf = regressor
    clf = clf.fit(train_x, train_y)
    models.append(clf)
for model in models:
    # train_y is logged so mae computes mae
    train_mae = mae(train_y, model.predict(train_x))
    val_mae = mae(val_y, model.predict(val_x))
    print('With model: {}\nTrain mae: {}\nVal. MAE: {}'.format(model, train_mae, val_mae))
"""

# random forest - model complexity experiments:  best {'max_features': 20, 'n_estimators': 100, 'max_depth': 2}
tuned_parameters = [{'max_depth':[2,5,10], 'n_estimators': [100, 500], 'max_features': [0.25, 0.5, 0.75]}]
from sklearn.model_selection import GridSearchCV

scores = ['neg_mean_absolute_error']

for score in scores:
    print("# Tuning Random Forest hyperparameters for %s" % score)
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(tune_train[:10000,], tune_y[:10000])

    print("Best hyperparameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = tune_y, clf.predict(tune_train)
    tune_mae = mae(y_true, y_pred)   
    print(train_mae)
    
y_true, y_pred = y, clf.predict(train_attr)    
train_mae = mae(y_true, y_pred)
print(train_mae)




"""

tuned_parameters = [{'max_depth':[2], 'n_estimators': [100, 500, 1000], 'max_features': [10, 20, 30, 40, 50, 60, len(X.columns)]}]

scores = ['recall']

for score in scores:
    print("# Tuning Random Forest hyperparameters for %s" % score)
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best hyperparameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

"""

########## learing curve of RF

tune_train = train_attr[:20000000,]
tune_y = y[:20000000]

RANDOM_STATE = 42
# random forest - learning curve experiments
def randomforest_pred(X_train, Y_train, X_test):
	randomforest = RandomForestRegressor(random_state=RANDOM_STATE)
	randomforest.fit(X_train, Y_train)
	return randomforest.predict(X_test)

def mae(y_true, y_pred):
    return np.mean(np.absolute(y_pred - y_true), axis=-1)


size_lst = []
mae_lst=[]
#train_size = np.arange(0.05, 1.05, 0.2)
train_size = [0.05, 0.5, 0.75, 0.9, 1]
for size in train_size:
    X_train, X_test, Y_train, Y_test = train_test_split(tune_train, tune_y, test_size=0.2, random_state=RANDOM_STATE)
    n = int(size*len(X_train))
    X_train = X_train[:n,:]
    Y_train = Y_train[:n]
    Y_pred = randomforest_pred(X_train,Y_train,X_test)
    mae1 = mae(Y_test, Y_pred)
    size_lst.append(n)
    mae_lst.append(mae1)
    
test_result_rand = pd.DataFrame(np.column_stack([size_lst, mae_lst]), 
                               columns=['sample_size', 'mae'])
# plot 
fig1 = plt.figure(figsize=(8, 6), dpi=80)                                                                                           
ax1 = fig1.add_subplot(111)                                                                                                                                                           
ax1.plot(test_result_rand.sample_size, test_result_rand.mae) 
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('MAE')
plt.title('Effect of training set size on MAE')
plt.show()


# random forest - model complexity experiments:  n_estimators
tuned_parameters = [{'n_estimators': [1,10,20,30,40,50]}]
scores = ['neg_mean_absolute_error']

n_lst = [1,10,20,30,40,50]
mae_lst=[]
for score in scores:
    print("# Tuning Random Forest hyperparameters for %s" % score)
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(tune_train[:10000,], tune_y[:10000])

    print("Best hyperparameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        mae_lst.append(-mean)

# plot 
fig1 = plt.figure(figsize=(8, 6), dpi=80)                                                                                           
ax1 = fig1.add_subplot(111)                                                                                                                                                           
ax1.plot(n_lst, mae_lst) 
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('MAE')
plt.title('Effect of Number of Trees on MAE')
plt.show()
    
# random forest - model complexity experiments:  max_depth
tuned_parameters = [{'max_depth': [1,2.5,5,7.5,10,15,20]}]
scores = ['neg_mean_absolute_error']

n_lst = [1,2.5,5,7.5,10,15,20]
mae_lst=[]
for score in scores:
    print("# Tuning Random Forest hyperparameters for %s" % score)
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(tune_train[:10000,], tune_y[:10000])

    print("Best hyperparameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        mae_lst.append(-mean)

# plot 
fig1 = plt.figure(figsize=(8, 6), dpi=80)                                                                                           
ax1 = fig1.add_subplot(111)                                                                                                                                                           
ax1.plot(n_lst, mae_lst) 
plt.legend()
plt.xlabel('Max Tree Depth')
plt.ylabel('MAE')
plt.title('Effect of Tree Depth on MAE')
plt.show()





submission = pd.concat([test['row_id'], pd.DataFrame(pred_rt, columns=['duration'])], axis=1)
submission.to_csv('submission_rf.csv',index=False)



pred_xgb = model_xgb.predict(dtest)
pred_xgb = np.exp(pred_xgb)
print('Test shape OK.') if test.shape[0] == pred_xgb.shape[0] else print('Oops')
pred_xgb

submission = pd.concat([test['row_id'], pd.DataFrame(pred_xgb, columns=['duration'])], axis=1)
submission.to_csv('submission-xgb.csv',index=False)


df = pd.read_csv('submission-xgb.csv')




