# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:40:51 2021

@author: Yufeng
"""
from typing import List
from typing import Dict

import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error



def get_test_days(df: pd.DataFrame,
                 n_days: int,
                 sample_n_dates: int,
                 train_validity: float = 0.8) -> Dict:
    '''
    given a sorted dataframe, get dates for each id that are eligible for training
    '''
    
    # days that have eligible data within 14 days ahead
    
    # get dates from each thermostat id
    user_list = df['Identifier'].unique()
    
    # set df's index as timestamp column
    df = df.set_index('lastReadingTimestamp').copy()
    
    # get test_days
    
    test_days = {}
    for user_id in user_list:
        
        # get data from one user id
        df_user = df[df['Identifier'] == user_id]
        
        # number of samples in test day and training day, respectively
        df_user_day = df_user.groupby(pd.Grouper(freq='D')).agg({'actualTemperature': 'count'})
        df_user_day = df_user_day.rename(columns={'actualTemperature': 'test_day_count'})
        
        # training day counts
        df_user_day['train_day_count'] = df_user_day.rolling('{}D'.format(n_days+1), min_periods=n_days+1).sum()
        
        # filter invalid days out
        df_user_day = df_user_day[
            (df_user_day['test_day_count'] >= 288-8) &\
            (df_user_day['train_day_count'] >= 288* n_days* train_validity)
        ]
        
        if len(df_user_day) == 0:
            test_date = []
        
        elif sample_n_dates != 0:
            test_date = np.random.choice(list(df_user_day.index), sample_n_dates)
            
        else:
            test_date = list(df_user_day.index)
            
        test_days.update({user_id: test_date})
        
    return test_days




class Indoor_temp_pred:
    
    '''
    A common class to train and predict indoor temperature
    
    Input:
    
    - df: Index should not be datetime
    - feature_col: Should include label as well
    - hist_term: if 0, only used t-1 as features, else, use t-1 to t-n as features
    
    Attributes:
    
    - pred_set: includes init_data
    
    Guidance:
    - First, after defining the class, input a time interval that needs to be tested/predicted
    - Second, get training set
    - Train the model, check whether the mse acceptable
    - If yes, predict the model. For testing, a test mse can be generated.
    
    '''
    
    def __init__(
        
        self,
        df: pd.DataFrame,
        feature_col: List,
        hist_term: int,
        n_train_days: int,
        label: str,
        
    ):
        
        self.user_id = self._get_user_id(df)
        self.feature_col = feature_col
        self.data = self._init_df(df)
        self.label = label
        self.hist_term = hist_term # should be min/5min
        self.n_train_days = n_train_days
        self.trained = False
        
    
    def _init_df(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # set index
        df = df.set_index('lastReadingTimestamp')
        
        # select useful features and set uniform order
        df = df.reindex(columns=self.feature_col)
        
        return df
    
    def _get_user_id(self, df) -> int:
        
        user_id = df['Identifier'].unique()
        if len(user_id) != 1:
            raise ValueError('user id is not unique in given data set')
            
        else:
            return user_id[0]
        
    def get_start_end_time(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        
        '''
        add datetime for prediction or test to the class
        '''
        self.start_ts = start_ts
        self.end_ts = end_ts
        
    def _get_hist_term(self, df : pd.DataFrame, for_pred: bool=False) -> pd.DataFrame:
    
        '''
        generate historical terms based on given data section
        '''
        df = df.copy()
        for col in self.feature_col:

            col_orig = df[col].copy() # keep original column

            # if col is label, don't drop, else drop the original values
            if col == self.label:
                pass
            else:
                df = df.drop(columns=col) # drop it temporarily

            for i in range(1, self.hist_term+2):

                # for each hist_term, add the shift of this column with lookback time
                df[f'{col}_{i*5}min'] = col_orig.shift(i)
                
        if for_pred:
            # if this fun is used to generate features for prediction, then drop the label
            df = df.drop(columns=self.label)
            
        # aftre adding the historical terms, drop nans
        df = df.dropna()

        return df
    
    # build training set based on given datetime
    def train_set_generator(self) -> pd.DataFrame:

        '''
        given user id and datetime, generate trainning set. 
        The date can be a test date, or the datetime of the start of a hold
        '''

        # select dates for training based on the given datetime and n_train_days
        df_user = self.data[(self.start_ts - pd.Timedelta(self.n_train_days, unit='day')) : (
            self.start_ts - pd.Timedelta(5, unit='min'))].copy()

        
        # find break points that longer than 15 min from the previous one
        continue_check = df_user.index.to_series().diff().gt(pd.Timedelta(15, unit='min'))
        continue_check = continue_check.cumsum() # indexing the sections

        n_continue = continue_check.groupby(continue_check).count()
        n_continue = n_continue[n_continue > self.hist_term]

        # get training data by section
        data_points = []
        for i in n_continue.index:

            subset_index = continue_check[continue_check == i].index
            subset = df_user.reindex(index=subset_index).copy()

            subset = self._get_hist_term(subset)       

            data_points.append(subset)   
        
        # add training set as an attribute
        data_points = pd.concat(data_points)
        self.train_set = data_points.sort_index()
        
        # add on columns for training
        self.use_col = [col for col in data_points.columns if col != self.label]
    
    
    def fit_model(self, model: str):
    
        
        '''
        Based on generated training set, train the model.
        Can choose the model type with paramter model.
        '''
        
        # reserve original y values
        y_orig = self.train_set[self.label]

        # standardize
        self.train_set_mean = self.train_set.mean()
        self.train_set_std = self.train_set.std()

        self.train_set_stand = self.train_set.subtract(
            self.train_set_mean).divide(self.train_set_std).copy()

        # check nan after standardize
        self.train_set_stand = self.train_set_stand.fillna(0)

        # define X and y
        X = self.train_set_stand[self.use_col].values
        y = self.train_set_stand[self.label].values.reshape(-1, 1)

        # train the model
        # choose the model
        if model == 'l2':
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], cv=5).fit(X, y)
        
        elif model == 'l1':
            clf = LassoCV(cv=5).fit(X,y)
            
        else:
            raise ValueError('model type is not recognized')
        
        # add train model as an attribute
        self.clf = clf
        
        # predict y using the entire training set
        y_pred = self.clf.predict(X)

        # invert transform predicted y
        y_pred = self.revert_T(y_pred)

        # calculate metric
        self.trained_mse = mean_squared_error(y_orig, y_pred)
        print(f'The best mse is {self.trained_mse}.')
        
        # add coefs
        self.coefs = self.clf.coef_
        self.trained = True
        
    def revert_T(self, t):
        '''
        stransform standardized temperature to the orignal scale
        '''
        t = (t * self.train_set_std[self.label]) + self.train_set_mean[self.label]
        
        return t
    
    
    def _get_pred_set(self, for_test: bool = True):
        
        '''
        Get dataset for prediction/test
        '''
        # build init data
        # start at 5 min before start date, and count backward
        init_data = self.data[ : (self.start_ts - pd.Timedelta(5, unit='min'))]
        init_data = init_data.iloc[-(self.hist_term + 1):]

        if init_data.index.to_series().diff().gt(pd.Timedelta(15, unit='min')).any():
            raise ValueError('Initial data not sufficient for prediction')

        # build pred data
        pred_data = self.data[self.start_ts : self.end_ts].copy()

        if pred_data.index.to_series().diff().gt(pd.Timedelta(15, unit='min')).any():
            raise ValueError('Prediction data has missing data longer than 15 min')


        if for_test:
            self.test_y_orig = pred_data[self.label].copy()

        # set pred labels as nan
        pred_data[self.label] = np.nan

        # concat init and pred data
        pred_set = pd.concat([init_data, pred_data])
        
        self.pred_data_index = pred_data.index
        self.debug_pred_set = pred_set.copy()
        
        return pred_set
        
        
    def predict_sequence(self, for_test: bool = True):
        
        '''
        Predict indoor temp based on trained model
        '''
        if self.trained:
            pass
        else:
            raise RuntimeError('model is not trained, please train the model first')
        
        # get pred_set
        pred_set = self._get_pred_set()
        
        # start predict
        for ts in self.pred_data_index:

            # generate features with init_data
            init_data = pred_set.loc[: ts]
            init_data = init_data.iloc[-(self.hist_term+2) :]
            
            
            feature = self._get_hist_term(init_data, for_pred=True)
            
            # standardize feature
            feature = feature.subtract(self.train_set_mean[self.use_col]).divide(
            self.train_set_std[self.use_col])
            feature = feature.fillna(0)
            
            if len(feature) != 1:
                print(type(feature))
                raise TypeError(
                    'feature has more than one record, feature not properly generated')
            
            # predict indoor temp
            #self.debug_feature = feature
            t_in = np.dot(feature.values, self.coefs.T)
            t_in = np.asscalar(t_in)
            
            # inverse transform
            t_in = self.revert_T(t_in)
            
            # add pred t_in on the pred_set
            pred_set.loc[ts, self.label] = t_in
            
        if pred_set.isna().any().any():
            raise ValueError('predict done, but nan values found')
        
        # add predicted results on
        self.pred_set = pred_set
        
        # add test mse if needed
        if for_test:
            y_pred = pred_set[self.start_ts: self.end_ts][self.label]
            
            self.test_mse = mean_squared_error(self.test_y_orig, y_pred)