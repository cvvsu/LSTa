import os, time, glob
import numpy as np
import pandas as pd 
from tqdm import tqdm 

import requests
from bs4 import BeautifulSoup

import geopandas as gpd

from multiprocessing.pool import ThreadPool as Pool 


def mkdirs(fp):
    fp = f'{fp}'
    if not os.path.exists(fp):
        os.makedirs(fp)


def get_filenames(year):
    res_each_year = requests.get(f'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/')
    html = BeautifulSoup(res_each_year.text, 'html.parser')
    names = []
    for item in html.find_all('a'):
        name = item.string
        if name.endswith('.csv'):
            names.append(name)
    return names


def download_data(year, name, savefp):
    try:
        url = f'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{name}'
        df = pd.read_csv(url)
        df.to_csv(os.path.join(savefp, name), index=False)
    except Exception:
        print(year, name, url)
        pass


def merge_one_year(fp, year, savefp):
    r"""
    merge all the station files in one year according to the file path of this year
    """
    files = glob.glob(os.path.join(fp, f'{year}')+'/*.csv')
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    dfs = pd.concat(dfs)
    dfs.to_csv(f'{savefp}/{year}.csv', index=False)


def F2C(x, digit=4):
    r"""
    Convert the degree F to degree C.
    """
    c = (x - 32.0) * 5 / 9.0
    return round(c, digit)


if __name__=='__main__':
    
    # download the datasets
    year_start = 2012
    year_end = 2017
    for year in range(year_start, year_end):
        savefp = f'datasets/met/gsod/{year}'
        mkdirs(savefp)

        names = get_filenames(year)
        lenx = len(names)
        print(f'In {year}, there are {lenx} csv files.')

        with Pool() as p:
            p.starmap(download_data, zip([year]*lenx, names, [savefp]*lenx))

    # merge the csv files
    len_years = len(range(year_start, year_end))
    with Pool() as p:
        p.starmap(merge_one_year, zip(['datasets/met/gsod']*len_years, range(year_start, year_end), ['datasets/met/gsod']*len_years))

    # merge all the csv files into one file
    dfs = []
    for file in tqdm(glob.glob(r'datasets/met/gsod/*.csv')):
        df = pd.read_csv(file)
        df['DATE'] = pd.to_datetime(df['DATE'])
        if df.duplicated().sum() > 0:
            print(file)
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(f'datasets/met/gsod/{year_start}_{year_end-1}_raw.csv', index=False)

    # check the final file
    df = pd.read_csv(f'datasets/met/gsod/{year_start}_{year_end-1}_raw.csv', na_values=[9999.9, 999.9, 99.99])
    df['STATION'] = df['STATION'].astype(str)
    
    # # rename the columns
    df = df.rename(columns={
        'TEMP': 'mean',       
        'MAX': 'max',
        'MIN': 'min',       
        'STATION': 'station', 
        'LATITUDE': 'latitude', 
        'LONGITUDE': 'longitude', 
        'ELEVATION': 'elevation', 
        'DATE': 'date'
    })

    df['year'] = df['date'].apply(lambda x: x[:4])

    df = df[['station', 'date', 'year', 'mean', 'min', 'max', 'elevation', 'latitude', 'longitude']]


    for item in ['mean', 'min', 'max']:
        df[item] = df[item].apply(lambda x: F2C(x))

    df = df.dropna(axis=0)
    df.to_csv('datasets/met/global_Ta.csv', index=False)

