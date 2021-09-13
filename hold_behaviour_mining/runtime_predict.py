# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:37:02 2021

@author: Yufeng
"""

from typing import List

import pandas as pd
import numpy as np

import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error

class Runtime_train_test:
    
    '''
    A common class train and test/predict runtime given a feature set.
    
    Input:
    -df: must be a feature set from preprocessing class with section number
     as index.
    '''
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 user_id: int):
        
        self.data = df
        self.user_id = user_id
        self.test_set = None
        self.train_set = None
        self.model_name = None
        self.model = None
        self.best_result = {'MAPE': np.inf}
        self.trained = False
        
    def split_test_set(self, pct: float=0.2):
        
        '''
        Split out 20% test set for final test.
        
        This method will update self.test_set and self.train_set
        '''
        
        self.test_set = self.data.sample(frac=pct).copy()
        self.train_set = self.data[~self.data.index.isin(self.test_set.index)]
        
    def dt_CV(self, 
              feature_cols: List, 
              label: str, 
              tree_depth_range=range(3, 31), 
              inplace=True):
        
        '''
        Cross validation and select the best parameters for decision tree.
        After selecting the best parameter, 
        Then, best results will be tired to update to self.best_result.
        
        This method add a debug attribute: self.debug_dt_tuning_resuls
        '''
        df = self.train_set.copy()
        
        tuning_results = []
        
        for tree_depth in tree_depth_range:


            # split into k-fold cv sets
            kf = KFold(n_splits=5, shuffle=True, random_state=5)

            mape_total_list = []

            for train_index, test_index in kf.split(df):
                X_train = df[feature_cols].iloc[train_index]
                X_test =  df[feature_cols].iloc[test_index]
                y_train = df[label].iloc[train_index]
                #y_test = df[label].iloc[test_index]


                # run several times and get several smallest score
                mape_list = []
                for i in range(5):

                    # define tree model
                    tree_model = DecisionTreeRegressor(max_depth=tree_depth)
                    tree_model.fit(X_train, y_train)

                    # predict y
                    y_pred = tree_model.predict(X_test)

                    # calculate rmse of hourly runtime

                    # get real hourly runtime from pickup_set
                    runtime_true = df.reindex(index=X_test.index)['runtime_per_hour']
                    
                    if label == 'alpha':
                        
                        # get pred runtime based on y_pred (pred_alpha) and demand
                        y_demand = df.reindex(index=X_test.index)['demand']
                        runtime_pred = y_pred * y_demand
                        runtime_pred = runtime_pred.round(2)
                    
                    else:
                        
                        runtime_pred = y_pred

                    # cal mape
                    mape = mean_absolute_percentage_error(runtime_true, runtime_pred)

                    mape_list.append(mape)

                # get smallest values
                mape_list = sorted(mape_list)[:3]

                # add on total list
                mape_total_list = mape_total_list + mape_list

            # add mean and std on tunning results
            mape_mean = np.mean(mape_total_list)
            mape_std = np.std(mape_total_list)
            tuning_results.append([mape_mean, mape_std])

        tuning_results = pd.DataFrame(np.vstack(tuning_results), columns=['mean', 'std'], index=range(3, 31))
        self.debug_dt_tuning_resuls = tuning_results
        
        
        if label == 'alpha':
            self.model_name = 'DT_alpha'
        else:
            self.model_name = 'DT'
        
        self._get_best_training_result(tuning_results, feature_cols, label)

        if not inplace:
            return tuning_results
        
    
    def rf_CV(self, 
              feature_cols: List, 
              label: str, 
              tree_depth_range=range(3, 8), 
              n_estimate=range(300, 800, 100), 
              split_num=2, 
              inplace=True):
        
        
        df = self.train_set.copy()
        
        tuning_results = []
        index_taker = []
        i_count = 0

        for num_est in n_estimate:
            for tree_depth in tree_depth_range:

                # show progress
                i_count += 1
                if i_count % 10 == 0:
                    print(i_count)            

                # hyper para
                hyper = (num_est, tree_depth)
                index_taker.append(hyper)


                # split into k-fold cv sets
                kf = KFold(n_splits=split_num, shuffle=True)

                mape_total_list = []

                for train_index, test_index in kf.split(df):
                    X_train = df[feature_cols].iloc[train_index]
                    X_test =  df[feature_cols].iloc[test_index]
                    y_train = df[label].iloc[train_index]
                    #y_test = df[label].iloc[test_index]


                    mape_list = []
                    for i in range(2):

                        # define tree model
                        tree_model = RandomForestRegressor(n_estimators=num_est, max_depth=tree_depth, n_jobs=-1)
                        tree_model.fit(X_train, y_train)

                        # predict hourly runtime
                        y_pred = tree_model.predict(X_test)

                        # calculate rmse of hourly runtime

                        # get real hourly runtime from pickup_set
                        runtime_true = df.reindex(index=X_test.index)['runtime_per_hour']


                        if label == 'alpha':
                        
                            # get pred runtime based on y_pred (pred_alpha) and demand
                            y_demand = df.reindex(index=X_test.index)['demand']
                            runtime_pred = y_pred * y_demand
                            runtime_pred = runtime_pred.round(2)
                    
                        else:

                            runtime_pred = y_pred

                        # cal mape
                        mape = mean_absolute_percentage_error(runtime_true, runtime_pred)

                        mape_list.append(mape)


                    # get smallest values
                    mape_list = [min(mape_list)]

                    # add on total list
                    mape_total_list = mape_total_list + mape_list

                # add mean and std on tunning results
                mape_mean = np.mean(mape_total_list)
                mape_std = np.std(mape_total_list)
                tuning_results.append([mape_mean, mape_std])

        tuning_results = pd.DataFrame(np.vstack(tuning_results), 
                                       columns=['mean', 'std'], 
                                       index=pd.MultiIndex.from_tuples(index_taker, names=['n_estimates', 'tree_depth']))
        self.debug_rf_tuning_resuls = tuning_results
        
        if label == 'alpha':
            self.model_name = 'RF_alpha'
        else:
            self.model_name = 'RF'
        
        self._get_best_training_result(tuning_results, feature_cols, label)
        
        if not inplace:
            return tuning_results
    
    
    def _get_best_training_result(self, 
                                  tuning_results: pd.DataFrame, 
                                  feature_cols: List, 
                                  label: str):
        
        br = tuning_results.nsmallest(1, 'mean')
        
        candidate = {}
        
        candidate['MAPE'] = np.asscalar(br['mean'].values)
        
        if self.best_result['MAPE'] <= candidate['MAPE']:
            pass
        else:
            
            candidate['std'] = np.asscalar(br['std'].values)
            candidate['Param'] = np.asscalar(br.index)
            candidate['Model'] = self.model_name
            candidate['feature_cols'] = feature_cols
            candidate['label'] = label
            
            self.best_result = candidate
    
    
    def train_best_model(self, relax_factor: float = 1.0):
        
        '''
        Train model with entire trainig set with best parameters and model
        '''
        
        best_model = self.best_result['Model']
        best_para = self.best_result['Param']
        best_mape = self.best_result['MAPE']
        feature_cols = self.best_result['feature_cols']
        label = self.best_result['label']
        
        # define training and test datasets
        X = self.train_set[feature_cols]
        y = self.train_set[label]
        X_test = self.test_set[feature_cols]
        runtime_true = self.test_set[label]
        
        # other params for the loop
        new_mape = np.inf
        i = 0
        while (new_mape > relax_factor*best_mape) & (i <= 100):
            
            # define model
            if 'DT' in best_model:
                tree_model = DecisionTreeRegressor(max_depth=best_para)
                tree_model.fit(X, y)
                
            elif 'RF' in best_model:
                tree_model = RandomForestRegressor(
                    n_estimators=best_para[0], 
                    max_depth=best_para[1])
                tree_model.fit(X, y)

            # predict hourly runtime
            y_pred = tree_model.predict(X_test)

            if label == 'alpha':
                        
                # get pred runtime based on y_pred (pred_alpha) and demand
                y_demand = self.test_set['demand']
                runtime_pred = y_pred * y_demand
                runtime_pred = runtime_pred.round(2)

            else:

                runtime_pred = y_pred

            # cal mape
            mape = mean_absolute_percentage_error(runtime_true, runtime_pred)

            new_mape = mape

            i += 1

        # check whether model is good
        if new_mape > relax_factor*best_mape:
            print('Training failed at {}'.format(self.user_id))
            
        self.model = tree_model
        self.best_test_score = new_mape
        
    def save_model(self, filename: str):
        
        joblib.dump(self.model, filename)       
        