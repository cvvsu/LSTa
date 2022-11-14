import os
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse

from scipy.spatial.distance import cdist
from matplotlib.colors import Normalize
from multiprocessing import Pool


def get_interpolation(X, Y, vals, num_rows, num_cols):
    """
    Interpolate the measured points to grids.

    Parameters:
        X (ndarray) -- [N, 2], (x, y) locations with known values
        Y (ndarray) -- [M, 2], (x_, y_) locations with unknown values
    """
    assert len(X) == len(vals)
 
    # get the Euclidean distance between X and Y
    dist = cdist(X, Y, 'euclidean')

    # get the weights (inverse distance)
    # in case the 0 distance, add a epsilon
    weights = 1.0 / (dist + 1e-6)

    # get the intepolation results
    res = np.sum(vals.reshape(-1, 1) * weights, axis=0) / (weights.sum(axis=0))

    # replace the values where the distance is 0
    # idx = np.where(weights > 1)[1]
    idx = np.where(dist == 0)[1]
    res[idx] = vals

    return res.reshape(num_rows, num_cols)


def interp_single_day(day):
    # global df
    # global Y
    # global num_rows
    # global num_cols

    df_ = df.loc[day]

    X = df_[['row', 'col']].values            

    results = []
    for item in ['mean', 'min', 'max']:
        z = df_[item].values
        result = get_interpolation(X, Y, z, num_rows, num_cols)
        results.append(result)
    
    # add a mask layer
    temp_mask = np.zeros((num_rows, num_cols))
    temp_mask[X[:, 0], X[:, 1]] = 1
    
    results = np.dstack(results + [temp_mask])
    savename = day.replace('-', '_')
    np.save(f'datasets/Ta/{savename}.npy', results)


if __name__ == '__main__':
    """
    Station `KANKAANPAA NIINISALO AIRFIELD, FI`
    and station `KANKAANPAA NIINISALO PUOLVOIM, FI`
    have the same longitudes and latitudes, and similar recorded temperatures.
    In this case, we keep the measurements from the `KANKAANPAA NIINISALO PUOLVOIM, FI` 
    station, since it has more record items.
    """

    # read the csv file
    if not os.path.exists('datasets/met/Ta_FI.csv'):
        df = pd.read_csv('datasets/met/Ta_FI_raw.csv', index_col=['date'])
        df = df[~(df['state'] == 'KANKAANPAA NIINISALO AIRFIELD, FI')]
        # df.to_csv('datasets/met/Ta_FI.csv')

        # read the longitude and latitude
        df_lon_lat = df[['longitude', 'latitude']].drop_duplicates()

        # read the tif file to obtain indexes of rows and cols
        data = rio.open('datasets/2010_01_01.tif')
        num_rows, num_cols = data.read(1).shape

        # get the longitude-latitude and row-col mapping 
        lon_lat_row_col = {}
        for lon, lat in df_lon_lat.values:
            row, col = data.index(lon, lat)
            k = str(lon) + '_' + str(lat)
            v = str(row) + '_' + str(col)
            lon_lat_row_col.update({k:v})

        # apply the mapping to the dataframe
        def get_row_col(x, y):
            row_col = lon_lat_row_col[str(x) + '_' + str(y)]
            row, col = row_col.split('_')
            return int(row), int(col)
        
        """
        `zip` method is much (4x) faster than `result_type`
        """

        # df[['row', 'col']] = df.apply(lambda x: get_row_col(x.longitude, x.latitude), axis=1, result_type='expand')
        # df['row'], df['col'] = zip(*df.apply(lambda x: get_row_col(x.longitude, x.latitude), axis=1))

        df['row'], df['col'] = zip(*df.apply(lambda x: get_row_col(x.longitude, x.latitude), axis=1))
        df.to_csv('datasets/met/Ta_FI.csv')
        
    else:
        df = pd.read_csv('datasets/met/Ta_FI.csv', index_col=['date'])

    # read the tif file to obtain indexes of rows and cols
    data = rio.open('datasets/2010_01_01.tif')
    num_rows, num_cols = data.read(1).shape
    y_, x_ = np.meshgrid(range(num_cols), range(num_rows))
    Y = np.hstack([x_.reshape(-1, 1), y_.reshape(-1, 1)])

    parser = argparse.ArgumentParser(description='XXX')
    parser.add_argument('--year', type=int, default=2010,
                        help='start year')

    args = parser.parse_args()

    days = [str(item)[:10] for item in pd.date_range(start=f'{args.year}-01-01', end=f'{args.year}-12-31', freq='1D')]
    
    lenx = len(days)

    os.makedirs('datasets/Ta', exist_ok=True)
    os.makedirs('datasets/Ta_mask', exist_ok=True)
    

    for ind, day in tqdm(enumerate(sorted(days))):
        # if ind < 10:
        df_ = df.loc[day]

        X = df_[['row', 'col']].values            

        results = []
        for item in ['mean', 'min', 'max']:
            z = df_[item].values
            result = get_interpolation(X, Y, z, num_rows, num_cols)
            results.append(result)
        
        # add a mask layer
        mask = np.zeros((num_rows, num_cols))
        mask[X[:, 0], X[:, 1]] = 1
        
        # normalize the values here
        results = np.dstack(results) / 50.0
        savename = day.replace('-', '_')
        np.save(f'datasets/Ta/{savename}.npy', results)
        np.save(f'datasets/Ta_mask/{savename}.npy', mask)

        ############################################################
        ############# NOTES             ############################
        # the saved mask also has stations outside Finland
        
        
        