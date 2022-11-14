import os, glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def F2C(x, digit=4):
    r"""
    Convert the degree F to degree C.
    """
    c = (x - 32.0) * 5 / 9.0
    return round(c, digit)


if __name__ == '__main__':

    dfs = []
  
    for year in range(2010, 2022):
        df = pd.read_csv(f'datasets/met/gsod/{year}.csv', na_values=[9999.9, 999.9, 99.99])


        # # rename the columns
        df = df.rename(columns={
            'TEMP': 'mean',       
            'MAX': 'max',
            'MIN': 'min',       
            'STATION': 'station', 
            'LATITUDE': 'latitude', 
            'LONGITUDE': 'longitude', 
            'ELEVATION': 'elevation', 
            'DATE': 'date',
            'NAME': 'state'
        })

        df = df[['date', 'state', 'latitude', 'longitude', 'elevation', 'station', 'mean', 'min', 'max']]
        
        df = df.dropna(how='any', axis='index')
        for item in ['mean', 'min', 'max']:
            df[item] = df[item].apply(lambda x: F2C(x))
        
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    ####################################
    # We use the stations not only in Finland
    ###################################
    
    df_FI = df[(df['latitude']>=59.80983161667775)&(df['latitude']<=70.07757531416388)&(df['longitude']>=19.511407971076007)&(df['longitude']<=31.57578223680118)]
    df_FI.to_csv('datasets/met/Ta_FI_raw.csv', index=False)

    print(df_FI.shape)
    print(df_FI.describe())
    print(df_FI.corr())
    

