import os, glob
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm
import argparse
from multiprocessing import Pool


def mkdirs(fp):
    os.makedirs(f'{fp}', exist_ok=True)


def rename(name):
    if len(name) == 8:
        return name[:4] + '_' + name[4:6] + '_' + name[6:]
    return name


def process_mask(qc, error=1):
    """
    1. https://blog.csdn.net/VictoriaLy/article/details/104302313
    2. https://en.wikipedia.org/wiki/Bit_numbering

    least significant bit (LSB): right bit
    most significant bit (MSB): left bit


    number: 00|01|00|01
    order:  76 54 32 10 
    """

    binary_repr_v = np.vectorize(np.binary_repr)
    mask_ = binary_repr_v(qc, 8)

    if error == 1:
        mask = np.char.endswith(mask_, '00') & np.char.startswith(mask_, '00')
    elif error == 3:
        # error < 3 K
        mask = np.char.startswith(mask_, '00') | np.char.startswith(mask_, '01') | np.char.startswith(mask_, '10')
    return mask


def process_lst(lst, qc, norm_value, na_value, error):

    # convert K to C; fill invalid values with np.nan
    lst = lst * 0.02 - 273.15
    lst[lst == -273.15] = np.nan

    # process mask
    mask = process_mask(qc, error)
    lst[~mask] = np.nan

    # normalize the lst data
    lst_norm = lst / norm_value

    return np.nan_to_num(lst_norm, nan=na_value), mask


def LST2npy(file, savefolder, norm_value=50, na_value=-2, error=1):

    name = file.split(os.sep)[-1].split('.')[0]

    # get related QC file
    QC_file = file.replace('LSTD', 'QCD') if 'LSTD' in file else file.replace('LSTN', 'QCN')

    # read the data and mask
    data = rio.open(file).read(1)
    mask = rio.open(QC_file).read(1)
    # print(data.shape, mask.shape)
    # print(file, QC_file)
    lst, qc = process_lst(data, mask, norm_value, na_value, error)
    # print(lst.shape, qc.shape)
    # print(lst.min(), lst.max(), qc.min(), qc.max())
    
    # save npy files
    folder_lst = os.sep.join(file.split(os.sep)[:-1]).replace('tif', f'{savefolder}')
    folder_qc = os.sep.join(QC_file.split(os.sep)[:-1]).replace('tif', f'{savefolder}')
    mkdirs(folder_lst)
    mkdirs(folder_qc)
    np.save(os.path.join(folder_lst, name+'.npy'), lst)
    np.save(os.path.join(folder_qc, name+'.npy'), qc)


if __name__ == '__main__':
    # download the datasets
    parser = argparse.ArgumentParser('GSOD')
    parser.add_argument('--lst_type', type=str, default='LSTN')
    args = parser.parse_args()

    lst_type = args.lst_type
    files = sorted(glob.glob(f'datasets/tif/{lst_type}/*.tif'))
    error = 1
    norm_value = 50
    na_value = -2
    lenx = len(files)

    
    with Pool() as p:
        p.starmap(LST2npy, zip(files, ['npy']*lenx, [norm_value]*lenx, [na_value]*lenx, [error]*lenx))