# -*- coding: utf-8 -*-
"""
Created on June 12 20:22:57 2021

@author: Yufeng
"""

import pandas as pd


class Rules_combiner:
    
    def __init__(self, df: pd.DataFrame):
        
        self.data = df.copy()
        self.back_up = df.copy()
        self.log = {}
        self.user_list = df['Identifier'].unique()
        self.user_iter = iter(self.user_list)
        #self.col_names = df.columns
        self.current_user_data = None
        self.current_user = None
        self.cons_iter = None
        self.current_cons = None
        
    def get_rules(self):
        
        rules = self.current_user_data[self.current_user_data['consequents'] == self.current_cons]
        
        for index, row in rules['antecedents'].items():
            print('{}: {}'.format(index, sorted(row)))
    
    def get_current_user_data(self, init_cons=True):
        
        self.current_user_data = self.data[self.data['Identifier'] == self.current_user]
        
        if init_cons:
            # update consequent list
            cons_list = self.current_user_data['consequents'].unique()
            self.cons_iter = iter(cons_list)
            
        
    def update_current_user(self):
        
        try:
            self.current_user = next(self.user_iter)
        except:
            print('The last user already')
        
        # configure cons_iter
        self.cons_iter = None
        
        # configure log
        self.log.update({self.current_user: []})
    
    def update_current_cons(self):
        
        try:
            self.current_cons = next(self.cons_iter)
        except:
            print('The last consequent already')
        
    
    def combine_rules(self, index_list):
        
        '''
        Given index list, combine these rules together
        '''
        # get data
        df_sub = self.data.reindex(index=index_list).copy()
        
        # combine antecedents
        comb = [i for a in df_sub['antecedents'] for i in a]
        comb = frozenset(comb)
        
        # get consequents
        cons = df_sub['consequents'].unique()
        cons = cons.item()
        
        # get consequent support
        cons_supp = df_sub['consequent support'].unique()
        cons_supp = cons_supp.item()
        
        # calculate new support and confidence
        ante_supp = df_sub['antecedent support'].sum()
        supp = df_sub['support'].sum()
        conf = supp/ante_supp
        
        # replace old rules
        new_rule = [self.current_user, comb, cons, ante_supp, cons_supp, supp, conf]
        self.data.loc[index_list[0]] = new_rule
        
        # drop other rules
        self.data = self.data.drop(index=index_list[1:])
        
        self.update_log(index_list)
        
    def back_up_data(self):
        
        self.back_up = self.data.copy()
        
    def reverse_data(self):
        
        self.data = self.back_up.copy()
        
    def update_log(self, index_list):
        
        self.log[self.current_user].append(index_list)
        