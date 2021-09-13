# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:41:06 2021

@author: Yufeng
"""

from typing import List
from typing import Dict

import pandas as pd
import numpy as np

from scipy import stats

from pvlib import location
from pvlib import irradiance

'''
NOTE: need to import parent class Runtime_control_period!
'''

class Runtime_pred_preprocessor(Runtime_control_period):
    
    '''
    Based on controled period generator, generate all needed features for prediction
    
    All data should be indexed with datatime, and removed outliers
    
    Input:
        -Outdoor temp: should be interpolated into 5-min interal first.
         And the column should be named as 'outdoor_temp'
        -solar data should be 5-min interval data wtih DHI, GHI, DNI, albedo
    '''
    
    def __init__(
        self,
        df: pd.DataFrame,
        outdoor_temp: pd.DataFrame,
        solar_data: pd.DataFrame,
        window_dir: str,
        azm_dict: Dict = {'N':0 ,'S':180 ,'W':270 ,'E':90 },
        init_cols: List = ['actualTemperature','actualHumidity', 'desiredHeat', 'desiredCool', 'coolRuntime','heatRuntime', 'hvacMode']):
        
        super().__init__(df, init_cols)
        
        self.outdoor = outdoor_temp
        self.solar = solar_data
        self.window_dir = window_dir
        self.train_set = None
        self.azm_dict = azm_dict
        
    def add_outdoor_temp(self):
        
        if self.train_set is None:
            
            raise RuntimeError('Please generate trainning set first')
        
        # round temperature for 2 digits
        self.train_set = self.train_set.join(self.outdoor)
        
    def add_solar(self):
        
        if self.train_set is None:
            
            raise RuntimeError('Please generate trainning set first')
        
        # orientation of user's window
        if len(self.window_dir) == 1:
            surface_az = self.azm_dict[self.window_dir]
            gvi = self._get_solar_GVI(self.solar, surface_az)
        
        # if the suite in at the corner, then add the max irradiation
        elif len(self.window_dir) == 2:
            surface_az_1 = self.azm_dict[self.window_dir[0]]
            surface_az_2 = self.azm_dict[self.window_dir[1]]

            gvi_1 = self._get_solar_GVI(surface_az_1)
            gvi_2 = self._get_solar_GVI(surface_az_2)

            gvi = pd.concat([gvi_1, gvi_2], axis=1).max(axis=1)

        # de localize
        gvi.index = gvi.index.tz_localize(None)
        gvi = gvi
        gvi.name = 'solar'

        self.train_set = self.train_set.join(gvi)
        
    
    def _get_solar_GVI(self, sur_az: int) -> pd.Series:
    
        # geo info
        lat, lon = 43.46, -79.17
        tz = 'EST'

        # Localize time index
        solar = self.solar.copy()
        solar.index = solar.index.tz_localize('EST')

        # creat location
        site = location.Location(lat, lon, tz=tz)

        # Solar position
        times = solar.index
        solar_position = site.get_solarposition(times=times)

        # POA irradiance
        poa_irr = irradiance.get_total_irradiance(surface_tilt=90,
                                                 surface_azimuth=sur_az,
                                                 dni=solar['DNI'],
                                                 ghi=solar['GHI'],
                                                 dhi=solar['DHI'],
                                                 solar_zenith=solar_position['apparent_zenith'],
                                                 solar_azimuth=solar_position['azimuth'],
                                                 albedo=solar['Surface_Albedo'])

        return poa_irr['poa_global']
    

    def _degree_hour(self, sect: pd.DataFrame, n: int) -> float:
        
        '''
        Calculate degree hour for a specific section/segment
        '''
        
        
        delta_T = n * (sect['actualTemperature'] - sect['outdoor_temp'])

        d_h = delta_T.sum() * 5 / 60

        return round(d_h, 2)
    
    
    def _runtime_hour(self, sect: pd.DataFrame, runtime: pd.Series) -> Dict:
        
        '''
        Calculate runtime and runtime/hour, return both values in a dict.
        '''

        # calculate runtime/hour
        length = len(sect)/12 # (hours)
        
        results_real = runtime.sum()/60 # trans sec to min

        resutls = (results_real/length) 

        return {'real': round(results_real, 2), 'norm': round(resutls, 2)}
    
    # calculate alpha, time, solar and so on
    def _feature_calculator(self, 
                           sect_name: float, 
                           return_dict=False):

        sect = self.train_set[self.train_set['cumsum'] == sect_name]
        
         # runtime mode
        mode =  sect['modeRuntime'].unique()
        if len(mode) > 1:
            raise ValueError('Runtime mode not uniform in section {}'.format(sect_name))
        elif mode[0] == 'heat':
            runtime = sect['heatRuntime']
            n = 1
        elif mode[0] == 'cool':
            runtime = sect['coolRuntime']
            n = -1
        else:
            print('Runtime mode is neither heat nor cool in section {}'.format(sect_name))
            raise ValueError('Runtime mode is neither heat nor cool in section {}'.format(sect_name))


        # alpha, demand, and runtime
        runtime_dict = self._runtime_hour(sect, runtime)
        runtime_per_hour = runtime_dict['norm']
        hvac_demand = self._degree_hour(sect, n)

        if hvac_demand ==0:
            print('degree_hour is zero in section {}'.format(sect_name))
            raise ValueError('degree_hour is zero in section {}'.format(sect_name))

        alpha = runtime_per_hour / hvac_demand
        alpha = round(alpha, 2)

        runtime = runtime_dict['real']

        # solar
        solar_mean = round(sect['solar'].mean(), 2)
        solar_std = round(sect['solar'].std(), 2)

        # month
        month = sect.index.month
        month = stats.mode(month)[0][0]

        # day of week
        day = sect.index.dayofweek
        day = stats.mode(day)[0][0]
        if day in [5,6]:
            day = 0
        else:
            day = 1

        # time of day
        time = sect.index.to_series().quantile(.5)
        date = time.date()
        time = time - pd.Timestamp(date)
        time = time.total_seconds()/3600 # hours from 0 a.m.
        time = round(time, 2)

        # outdoor temperature
        outTemp_mean = round(sect['outdoor_temp'].mean(), 2)
        outTemp_std = round(sect['outdoor_temp'].std(), 2)

        if return_dict:

            return {'alpha':alpha, 'solar_mean':solar_mean, 'solar_std':solar_std,
                    'outdoor_temp_mean':outTemp_mean, 'outdoor_temp_std':outTemp_std,
                    'month':month, 'dayOfweek': day, 'time': time,
                   'demand': hvac_demand, 'runtime_per_hour': runtime_per_hour, 'runtime': runtime}   
        else:
            return [alpha, solar_mean, solar_std,
                    outTemp_mean, outTemp_std, 
                    month, day, time, hvac_demand, 
                    runtime_per_hour, runtime]
        
    
    def feature_generator(self):
        
        '''
        Put all features into a dataframe.
        Add attributes self.feature_set, self.feature_cols
        '''
        data_taker=[]
        index_taker=[]

        for i in self.sect_index:

            # calculate features
            try:
                feature = self._feature_calculator(i)
            
            except:
                print('The section {} failed to generate features'.format(i))
                continue

            data_taker.append(feature)

            # add on index
            index_taker.append(i)

        data_taker = np.vstack(data_taker)
        
        self.feature_cols = ['alpha', 'solar_mean', 'solar_std', 
                             'outdoor_temp_mean', 'outdoor_temp_std', 
                             'month', 'dayOfweek', 'time', 
                             'demand', 'runtime_per_hour', 'runtime']
        
        self.feature_set = pd.DataFrame(data_taker, index=index_taker, 
                                   columns=self.feature_cols)
        
    def generate_feature_set(self):
        
        '''
        Wrap up all the methods related to feature set generation
        '''
        
        self.add_outdoor_temp()
        
        self.add_solar()
        
        self.feature_generator()