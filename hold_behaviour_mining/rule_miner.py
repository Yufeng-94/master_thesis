# -*- coding: utf-8 -*-
"""
Created on June 12 20:06:59 2021

@author: Yufeng
"""

import re
import numpy as np
import pandas as pd
from scipy import stats

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


class Rule_miner:
    
    
    '''
    A common class to find non-redundant rules for a specific user id
    
    Input:
    -df: The index should not be time-like. It should be event_list_June22.
     start and end time should be parsed into datetime-like types.
    -min_appear: The time at least the itemset appears to be considered as
     frequent itemset. It will be used to calculate support.
    '''
    
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 min_appear: int=10, 
                 conf: float=0.80, 
                 use_col = ['startTimestamp', 'endtTime', 'length',
                           'coolHoldTemp', 'heatHoldTemp', 'Setpoint_before_C',
                           'Setpoint_before_H', 'Setpoint_after_C', 'Setpoint_after_H', 'Mode',
                           'Connect_before', 'Connect_after'], 
                 feature_col = ['startTimestamp', 'endtTime', 'length',
                           'coolHoldTemp', 'heatHoldTemp','Mode',
                           'Connect_before', 'Connect_after', 'Heating_Setback', 
                              'Cooling_Setback', 'Year_Month']):
        
        self.use_col = use_col
        self.data = self._preprocess(df)
        self.user_id = self._get_user_id(df)
        self.min_appear = min_appear
        self.conf = conf
        self.feature_col = feature_col
        
        
    def _get_user_id(self, df):
        
        user_id = df['Identifier'].unique()
        
        if len(user_id) != 1:
            raise ValueError('User id not unique in this dataset.')
        else:
            return user_id.item()
        

    def _preprocess(self, df):
        
        df = df.copy()
        
        # add length column with unit=hour
        def length(row):
            
            time_diff = row['endtTime'] - row['startTimestamp']
            length = time_diff.total_seconds()/3600
            
            return length
            
        df['length'] = df.apply(length, axis=1)
        
        # reindex
        df = df.reindex(columns=self.use_col)
        
        return df
    
        
    def feature_extractor(self): # need df_all with datetime index
    
        index_taker = []
        values_taker = []
        for index, row in self.data.iterrows():

            
            # sp change
            before_Heatsp_change = round(row['heatHoldTemp'] - row['Setpoint_before_H'], 1)
            before_Coolsp_change = round(row['coolHoldTemp'] - row['Setpoint_before_C'], 1)
            
            # year and month
            y_m = row['startTimestamp'].strftime('%Y-%m')

            values_taker.append(np.array([before_Heatsp_change, before_Coolsp_change, y_m]))

            index_taker.append(index)

        self.temp_change =  pd.DataFrame(np.vstack(values_taker), 
                                           index=index_taker, 
                                           columns=['Heating_Setback', 'Cooling_Setback', 'Year_Month'])
        
        self.events_feature = self.data.join(self.temp_change).copy()
        
    
    def binning_features(self):
        
        # create labels
        label_1 = ['{0}-{1}'.format(i, i+1) for i in range(0, 4)]
        label_1.append('>=4')

        label_2 = ['{0}-({1})'.format(i, i-1) for i in range(-3, 1)]
        label_2.insert(0, '<=(-4)')

        # create bins
        bins_1 = list(range(0,5))
        bins_1.append(np.inf)

        bins_2 = list(range(-4, 1))
        bins_2.insert(0, -np.inf)

        for col in ['Heating_Setback', 'Cooling_Setback']:
            test = self.events_feature[col].astype('float')
            test_1 = test[test>=0]
            test_2 = test[test<0]

            total = pd.cut(test_1, bins= bins_1, right=False, labels=label_1).append(
            pd.cut(test_2, bins = bins_2, labels=label_2))

            total = total.sort_index()


            self.events_feature[col] = total

        # categorize length
        labels = ['<=2_Hours', '2-4_Hours', '4-6_Hours', '6-12_Hours', '>12_Hours']
        bins = [0, 2, 4, 6, 12, np.inf]

        
        self.events_feature['length'] = pd.cut(self.events_feature['length'], bins = bins, labels = labels)

        # categorize stp
        bins_heat = [0, 20, 22, 24, 26, 40]
        labels_heat = ['<=20', '20-22', '22-24', '24-26', '>26']

        bins_cool = [0, 22, 24, 26, 28, 50]
        labels_cool = ['<=22', '22-24', '24-26', '26-28', '>28']

        self.events_feature['heatHoldTemp'] = pd.cut(self.events_feature['heatHoldTemp'], bins=bins_heat, labels = labels_heat)
        self.events_feature['coolHoldTemp'] = pd.cut(self.events_feature['coolHoldTemp'], bins= bins_cool, labels = labels_cool)

        # categorize time of start and end
        labels_time = ['{0}:00-{1}:00'.format(i, i+2) for i in range(6, 23, 2)]
        labels_time.insert(0, '0:00-6:00')

        bins_time = pd.date_range(pd.to_datetime('06:00:00'), pd.to_datetime('23:55:00'), freq='2H', closed=None).tolist()

        bins_time.insert(0, pd.to_datetime('00:00:00'))
        bins_time.append(pd.to_datetime('23:59:00'))

        for col in ['startTimestamp', 'endtTime']:

            time_version = pd.to_datetime(self.events_feature[col].dt.strftime('%H:%M:%S'))

            self.events_feature[col] = pd.cut(time_version, bins=bins_time, labels=labels_time, right=False)
            
        
        self.length_label = labels
        self.heat_spt_label = labels_heat
        self.cool_spt_label = labels_cool
        self.pos_change_label = label_1
        self.neg_change_label = label_2
        self.time_label = labels_time
        
    
    def feature_set_generator(self):
        
        def add_preffix(row, name):
            return name+'&' + row
        
        self.feature_set = self.events_feature[self.feature_col].copy()
        
        for col in self.feature_col:

            self.feature_set[col] = self.feature_set[col].apply(add_preffix, name=col)
            
        self.feature_set = self.feature_set.dropna()
            
            
    def rule_generator(self):
        
        # to see if there are enough samples
        if len(self.feature_set) < 50:
            
            raise ValueError('Too less sample for {}'.format(self.user_id))
        
        TE = TransactionEncoder()
        te = TE.fit(self.feature_set.values).transform(self.feature_set.values)
        df_te = pd.DataFrame(te, columns=TE.columns_)

        # frequent patterns
        supp = self.min_appear/len(self.feature_set)
        fp = apriori(df_te, min_support=supp, use_colnames=True)
        self.frequent_pattern = fp
        
        # rules
        rules = association_rules(fp)

        rules = rules[['antecedents', 'consequents', 'antecedent support',
               'consequent support', 'support', 'confidence', 'lift']]

        rules = rules[(rules['support'] >= supp) &\
            (rules['confidence'] >= self.conf)]

        self.bebug_rules = rules
        self.rules = rules.copy()
        
        # remove redundant rules
        self._rule_engine()
        
    def _rule_engine(self):
        
        # rule engines 
        # mode must be in antecedents
        index_taker = []

        for index, values in self.rules['antecedents'].items():

            if any('Mode' in re.split('&', ele)[0] for ele in values ):

                index_taker.append(index)
            else:
                continue

        self.rules=self.rules.loc[index_taker]

        # cooling and heating can't be in a same rule
        index_taker = []

        for index, rows in self.rules.iterrows():

            values = [ele for ele in rows['antecedents']] + [ele for ele in rows['consequents']]

            cool = any(re.search('(C|c)ool', ele) for ele in values)
            heat = any(re.search('(H|h)eat', ele) for ele in values)

            if heat & cool: 
                continue

            else:

                index_taker.append(index)

        self.rules = self.rules.loc[index_taker]

        # consequents have to be 1 length
        length = self.rules['consequents'].apply(lambda x: len(x))
        self.rules = self.rules[length == 1]

        # antecedents have to be a set that has no subsets

        index_total = []
        for cons in self.rules['consequents'].unique():

            # find subgroups with the same consequents
            subgroup=self.rules[self.rules['consequents'] == cons]

            for index, row in subgroup['antecedents'].items():

                for index_c, row_c in subgroup['antecedents'].items():

                    if row == row_c:
                        continue

                    elif row.issuperset(row_c):
                        index_total.append(index)

                    else:
                        continue   

        self.rules = self.rules.drop(index=index_total)
        
        
    def generate_rules(self):
        
        '''
        A wrap-up function of all the other functions.
        Generate rules at one step
        '''
        
        self.feature_extractor()
        
        self.binning_features()
        
        self.feature_set_generator()
        
        self.rule_generator()