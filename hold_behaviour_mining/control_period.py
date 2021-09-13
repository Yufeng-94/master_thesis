# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:34:25 2021

@author: Yufeng
"""

from typing import List

import pandas as pd
import numpy as np


class Runtime_control_period:
    
    '''
    A common class used to preprocess prediction data (dataset only has 1 user).
    Only 
    
    Input:
    
    -df: Should have datetime index. 
         And the index should be well sorted.
         And outliers should be removed.
    
    '''
    
    def __init__(self,
                 df: pd.DataFrame,
                 init_cols: List):
        
        self.init_cols = init_cols
        self.user_id = self._get_user_id(df)
        self.data = self._preprocessor(df)
        
        
    
    def _preprocessor(self, df):
        
        '''
        Preprocess data
        '''
        
    # drop off mode
        df = df[df['hvacMode'] != 'off']
        
    # choose useful initial cols
        df = df.reindex(columns=self.init_cols)
        
    # add runtime type
        def runtime_type(row):
    
            '''If a sample with both heat and cool runtime, assgin this sample to heat runtime. 
            Only 6 samples with such case.'''

            if (row['coolRuntime'] + row['heatRuntime']) == 0:
                return np.nan

            elif row['heatRuntime'] != 0:
                return 'heat'
            else:
                return 'cool'
            
        runtime_mode = df.apply(runtime_type, axis=1)

        # fill in nan values
        while runtime_mode.isna().sum() != 0:
            runtime_mode = runtime_mode.fillna(method='ffill', limit=1)
            runtime_mode = runtime_mode.fillna(method='bfill', limit=1)
        # add on df
        df['modeRuntime'] = runtime_mode
        
    # add setpoint type
        def setpoint_type(row):
    
            if row['modeRuntime'] == 'heat':
                return row['desiredHeat']

            else:
                return row['desiredCool']

        setpoint = df.apply(setpoint_type, axis=1)

        # add control setpoint to training set
        df['setpoint'] = setpoint

        
        return df
        
    def _get_user_id(self, df):
        
        user_id = df['Identifier'].unique()
        
        if len(user_id) != 1:
            raise ValueError('User id not unique in this dataset.')
        else:
            return np.asscalar(user_id)
        
    def remove_free_floating(self):
        
        '''
        Remove the section where indoor temperature floats between cooling and 
        heating setpoints and no need to start AC for cooling or heating.
        
        This method adds self.train_set and self.non_run
        '''
        
        df = self.data
        
        # exclude samples with runtime
        non_run = df[(df['coolRuntime'] == 0) & (df['heatRuntime'] == 0)].copy()

        # numbering continous sections
        non_run['bool'] = non_run.index.to_series().diff().ne(pd.Timedelta(5, unit='min'))
        non_run['cumsum'] = non_run['bool'].cumsum()

        need_check = non_run.groupby('cumsum').count().iloc[:, 0]
        
        # only check continuous sections longer than 30 min (6 samples)
        # shorter ones are considered as training set directly
        need_check = need_check[need_check > 6]
        # sections shorter longer than 4 hours (48 samples) will be removed
        # as they're too long to be a controled section
        elim_sect = list(need_check[need_check > 48].index)
        need_check = list(need_check[need_check <= 48].index)
        
        # set cumsum as index
        non_run = non_run.reset_index()
        
        non_run.set_index('cumsum', inplace=True)
        
        # control section check
        # get past sections and eliminated sections
        self._check_control_sect(non_run, need_check, elim_sect)
        
        # timestamps of eliminated sections
        elim_index = non_run[non_run.index.isin(self.elim_sect)]['lastReadingTimestamp']
        self.train_set = df.drop(index=elim_index).copy()
        non_run = non_run.drop(index=self.elim_sect)
        self.non_run = non_run.copy()
        
        
    def _check_control_sect(self,
                            non_run: pd.DataFrame,
                            need_check: List,
                            elim_sect: List):
        
        '''
        Check whether a section is a controled section.
        
        -Alg for control periods' check:
            -set cumsum as index for convenient indexing
            -remove na and zero of 1st diff; zero causes confusions when judging signs
            -if the left elements are less than 5, then directly assign as a normal section
            -do 1st diff again and get cumsum, the tied numbers are the time of sign changes
            -number of sign change and its count (or length)
            -if count == 1, check the trend, and assgin a hvac mode
            -if count == 2, its a increase-decrease section, possible for both modes, thus assgin as normal
            -if count >2, then if only 2 of them are longer than 10 min, then the other shifts can be ignored.
             Or the section should be excluded.
            -if the 2 main shifts have the same trend, do the same choice when count ==1,
             else do when count ==2.
        '''
        
        past_sect = {}
        
        for i in need_check:

            # 1st diff if indoor temperature
            test_section = non_run.loc[i]['actualTemperature'].diff().dropna()

            # whether positive, and transfer bool to int after removing 'zero' differences
            test_section = test_section[test_section != 0]
            test_section = test_section.gt(0).astype('int')

            if len(test_section) < 5:
                # if too short after removing 0, just consider it as a normal section
                past_sect.update({i: 'normal'}) 
                continue

            # number of appeared signs
            num_signs = test_section.diff().ne(0).cumsum()

            # sign numbers and counts
            num_signs_unique = num_signs.groupby(num_signs).count()

            if len(num_signs_unique) == 1:

                # check sign
                if all(test_section == 1):
                    past_sect.update({i: 'cool'})

                elif all(test_section == 0):
                    past_sect.update({i: 'heat'})

                else:
                    raise ValueError(
                        'The signs are not identical in a single sign judgement for section {}'.format(i))

            elif len(num_signs_unique) == 2:

                past_sect.update({i: 'normal'})

            else:
                # change index: conveient for determining the signs of main shifts in the following process
                test_section.index = num_signs.values
                
                # it is ok if only two sign shifts longer than 10 min
                main_shift = (num_signs_unique > 2).sum()
                
                if main_shift > 2: # only two main shift is allowed
                    # eliminate this section
                    elim_sect.append(i)

                else:
                    # check the sign of two main shifts

                    # main shift numbers
                    main_sft_index = num_signs_unique[num_signs_unique > 2].index
                    
                    if len(main_sft_index) == 1:
                        if all(test_section[main_sft_index]):
                            past_sect.update({i: 'cool'})
                        else:
                            past_sect.update({i: 'heat'})

                    elif len(main_sft_index) == 2:

                        # check interity
                        if (len(test_section[main_sft_index[0]].unique()) != 1) |\
                        (len(test_section[main_sft_index[1]].unique()) != 1):
                            raise ValueError('Signs in main shifts are not identical for section {}'.format(i))

                        # True for increase, False for decrease
                        sec_1 = all(test_section[main_sft_index[0]])
                        sec_2 = all(test_section[main_sft_index[1]])

                        # check signs
                        if sec_1 + sec_2 == 2:
                            past_sect.update({i: 'cool'})

                        elif sec_1 + sec_2 == 0:
                            past_sect.update({i: 'heat'})

                        else:
                            past_sect.update({i: 'normal'})

                    elif len(main_sft_index) > 2:
                        raise ValueError('Number of main shifts is larger than 2 for section {}'.format(i))
                        
        self.past_sect = past_sect
        self.elim_sect = elim_sect
    
    def remove_open_ends(self):
        
        '''
        remove a candidate controled section has runtime on both ends
        '''
        # call the method
        self._check_both_ends()
        
        # timestamps of noRun_sect
        noRun_index = self.non_run[self.non_run.index.isin(self.noRun_sect)]['lastReadingTimestamp']

        # drop indices for both attributes
        self.train_set = self.train_set.drop(index=noRun_index)

        self.non_run = self.non_run.drop(index=self.noRun_sect)
        
    
    def _check_both_ends(self):
        
        '''
        check Whether a candidate controled section has runtime on both ends
        
        This method add an attribute self.noRun_sect
        
        -Alg:
            -get timestamps of both ends of a section. 
             Note the section number should be the one in the non_run,
             instead of past_sect.
            -If no runtime or runtimes are zero, add the section number to noRun_sect.
            -If the section not in the past_sect, it should be a short section
             that are directly considered as a controled period.
            -If the section type is inconsistent with runtime type,
             e.g. cool runtime vs heat section type. Add it to noRun_sect.
            -If section types at both ends are different, also add it to noRun_sect.             
        '''
        
        noRun_sect = []

        for i in self.non_run.index.unique():

            sect = self.non_run.loc[[i]] # to make sure result is dataframe


            # get the first and last timestamp
            t_min = sect['lastReadingTimestamp'].iloc[0]
            t_max = sect['lastReadingTimestamp'].iloc[-1]

            # get timestamps that a step forward and backward from the first and last timestamp
            t_min_f = t_min - pd.Timedelta(5, unit='min')
            t_max_b = t_max + pd.Timedelta(5, unit='min')

            # get runtime in original data
            try:
                runtime_f = self.train_set.loc[t_min_f].squeeze()
            except:

                noRun_sect.append(i)

                continue

            # runtime and type
            type_f = runtime_f['modeRuntime']
            runtime_f = runtime_f[['coolRuntime', 'heatRuntime']]


            try:
                runtime_b = self.train_set.loc[t_max_b].squeeze()
            except:

                noRun_sect.append(i)

                continue

            # runtime and type
            type_b = runtime_b['modeRuntime']
            runtime_b = runtime_b[['coolRuntime', 'heatRuntime']]


            # check if runtime at both ends not equal to zero
            if runtime_f.sum() * runtime_b.sum() == 0:

                noRun_sect.append(i)

                continue

            # whether section type consistent with runtime type

            # if not in past_sect, then it is the shorter section
            try:
                sect_type = self.past_sect[i]
            except:

                continue
                
            if type_f != type_b:

                noRun_sect.append(i)

                continue

            if sect_type == 'normal':

                continue

            elif type_f == sect_type:

                continue
                    
            elif type_f != sect_type:
                
                noRun_sect.append(i)

                continue
        # add to the attribute
        self.noRun_sect = noRun_sect
        
        
    def select_segments(self):
        '''
        Based on existed select segment, further select sections.
        This selection is based on the combination with 
        runtime and non runtime sections.
        
        Add attribute: self.total_check
        
        -Alg:
            -The time series has uniform user_ids
            -the t-s is continuous
            -Within which has the same kind of runtime mode
            -Setpoint for this kind of mode remain constant
            -remove sections shorter than 30 min
            -if longer than 4 hours, segment into 4 hours
        '''
        
        # boolean series based on the constraints above
        conti_check = self.train_set.index.to_series().diff().ne(pd.Timedelta(5, unit='min'))

        mode_check = self.train_set['modeRuntime'] != self.train_set['modeRuntime'].shift()

        setpoint_check = self.train_set['setpoint'].diff().ne(0)

        total_check = conti_check | mode_check | setpoint_check
        
        # add total to training set
        self.total_check = total_check

        self.train_set['cumsum'] = total_check.cumsum()
        
        # remove too short sections: threshold 0.5 hour
        sect_length = self.train_set.groupby('cumsum').count().iloc[:, 0]

        sect_notShort = sect_length[sect_length >= 6]

        self.train_set = self.train_set[self.train_set['cumsum'].isin(sect_notShort.index)]
        
        # remove sections without runtime at all
        check_whether_runtime = self.train_set.groupby('cumsum').sum()[['coolRuntime', 'heatRuntime']].sum(axis=1)
        check_whether_runtime = check_whether_runtime[check_whether_runtime > 0]

        self.train_set = self.train_set[self.train_set['cumsum'].isin(check_whether_runtime.index)]
        
        # cut the segments into shorter length
        self._split_segments()
        
        # remove zero runtime sections
        sum_runtime = self.train_set.groupby('cumsum').sum()[['coolRuntime', 'heatRuntime']].sum(axis=1)
        zero_runtime = sum_runtime[sum_runtime == 0]
        self.train_set = self.train_set[~self.train_set['cumsum'].isin(zero_runtime.index)]
        
        
        
    def _split_segments(self):
        
        '''
        cut segments into 4-hour length.
        
        -Alg:
            -if length < 48, pass
            -else, if the remainder (rem) less <= 12, then the last section is 48+rem
            -else, if the remainder > 12, the rem can be an individual section
        '''
        
        for i in self.train_set['cumsum'].unique():
    
            # extract section
            subset = self.train_set[self.train_set['cumsum'] == i]

            # compare length
            le = len(subset)

            if le <= 48:
                continue
            elif le % 48 <= 12:
                n = int(le / 48) -1
            else:
                n = int(le / 48)

            split_points = []
            # get index for the first timestamp in train_set

            s_1 = subset.iloc[0].name

            for j in range(n):

                # defined 2nd split timestamps
                s_2 = s_1 + pd.Timedelta(hours=4)

                # append split pairs 
                split_points.append([s_1, s_2])

                # update s_1
                s_1 = s_2 + pd.Timedelta(minutes=5)

            # split the section
            delta = 1 # add to the cumsum
            for p in split_points:

                self.train_set.loc[p[0]:p[1], 'cumsum'] = i + 0.01*delta
                delta += 1
        
    def generate_controled_train_set(self):
        
        '''
        Wrap up all methods related to training set generation.
        Add Attribute self.sect_index.
        '''
        
        self.remove_free_floating()
        
        self.remove_open_ends()
        
        self.select_segments()
        
        # check number of segments generated:
        # should be larger than 300
        sect_count = self.train_set['cumsum'].unique()
        if len(sect_count) < 300:
            
            raise ValueError('Train set does not have enough samples')
            
        self.sect_index = list(self.train_set['cumsum'].unique())