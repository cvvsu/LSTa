import os, glob
import numpy as np
import pandas as pd
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

# from utils.datasets import *
try:
    from .datasets_utils import *
except:
    from datasets_utils import *


class LSTaDataset(Dataset):
    def __init__(self, input_files, output_files, mask_files, mask, pad_size, apply_transform=False):
        self.input_files = input_files
        self.output_files = output_files
        self.pad_size = pad_size
        self.apply_transform = apply_transform
        self.mask_files = mask_files
        self.mask = mask

    def __getitem__(self, index):
        # get file paths
        input_file = self.input_files[index]
        output_file = self.output_files[index]
        mask_file = self.mask_files[index]

        # load the files
        X = load_file(input_file)
        Y = load_file(output_file)
        mask = np.load(mask_file) * self.mask
        if Y.ndim == 3:
            mask = np.dstack([mask]*3)

        # apply transformation if specified
        if self.apply_transform:
            X, Y, mask = pad_crop([X, Y, mask], self.pad_size)
            X, Y, mask = random_hflip([X, Y, mask])
            X, Y, mask = random_vflip([X, Y, mask])

        # convert numpy array to tensor
        X, Y, mask = to_tensor([X, Y, mask])        

        return X, Y, mask
    
    def __len__(self):
        return len(self.input_files)


def build_dataset(args):

    # get the file paths
    input_folders = args.input_folders
    output_folders = args.output_folders
    mask_path = args.mask_path

    if len(input_folders) == 1:
        input_files = [[file] for file in sorted(glob.glob(input_folders[0]+'/*.npy'))]
    else:
        input_files = list(zip(*[sorted(glob.glob(input_folder+'/*.npy')) for input_folder in input_folders]))

    if len(output_folders) == 1:
        output_files = [[file] for file in sorted(glob.glob(output_folders[0]+'/*.npy'))]
    else:
        output_files = list(zip(*[sorted(glob.glob(output_folder+'/*.npy')) for output_folder in output_folders]))

    mask_files = sorted(glob.glob(mask_path+'/*.npy'))

    # # to save time, only 1827 days are utilized
    n_days = 1827
    input_files = input_files[-n_days:]
    output_files = output_files[-n_days:]
    mask_files = mask_files[-n_days:]

    if args.verbose:
        print(len(input_files), len(output_files), len(mask_files))

    assert len(input_files) == len(output_files)
    assert len(input_files) == len(mask_files)

    # generate the mask for training, validation, and test sets
    df = pd.read_csv(args.station_loc)
    df = df.sample(frac=1, random_state=args.seed)
    arr_temp = np.load(input_files[0][0])
    h, w = arr_temp.shape[:2]
    shape = (h, w) 
    train_mask, val_mask, test_mask = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    lenx = len(df)
    len_test = int(args.test_ratio * lenx)
    len_val = int(args.val_ratio * lenx)
    len_train = lenx - len_val - len_test

    train_df, val_df, test_df = df.iloc[:len_train, :], df.iloc[len_train:len_train+len_val, :], df.iloc[len_train+len_val:, :]

    train_mask[train_df.row, train_df.col] = 1
    val_mask[val_df.row, val_df.col] = 1
    test_mask[test_df.row, test_df.col] = 1

    if args.verbose:
        print(input_files[:3], output_files[:3], mask_files[:3])
        print(train_mask.shape, val_mask.shape, test_mask.shape)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        x, y = np.where(train_mask==1)
        axes[0].scatter(x, y)
        x, y = np.where(val_mask==1)
        axes[1].scatter(x, y)
        x, y = np.where(test_mask==1)
        axes[2].scatter(x, y)
        fig.savefig('mask.png')

        print(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask))        

    train_set = LSTaDataset(input_files, output_files, mask_files, train_mask, args.pad_size, apply_transform=True)
    val_set = LSTaDataset(input_files, output_files, mask_files, val_mask, 0, apply_transform=False)
    test_set = LSTaDataset(input_files, output_files, mask_files, test_mask, 0, apply_transform=False)

    if args.dist == 'DDP':
        if os.environ['RANK'] == 0:
            logger.info(f'#Samples in training, validation, and test sets are {len(train_set)}, {len(val_set)}, and {len(test_set)}, respectively.')    

        # the whold batch_size is args.batch_size * world_size
        tr_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, drop_last=False, sampler=DistributedSampler(train_set))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=False, num_workers=args.num_workers, drop_last=False, sampler=DistributedSampler(val_set))
        # te_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    else:
        logger.info(f'#Samples in training, validation, and test sets are {len(train_set)}, {len(val_set)}, and {len(test_set)}, respectively.')    
        tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    te_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)

    return tr_loader, val_loader, te_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Test')
    parser.add_argument('--output_folders', nargs='+', default=['Ta'])
    parser.add_argument('--input_folders', nargs='+', default=['npy/LSTD', 'npy/LSTN'])
    parser.add_argument('--pad_size', type=int, default=20)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--dist', type=str, default='none')
    parser.add_argument('--mask_path', type=str, default='Ta_mask', help='masks used to calculated losses. LST2Ta')
    parser.add_argument('--station_loc', type=str, default='station_loc.csv', help='rows and cols for all stations')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    tr_loader, val_loader, te_loader = build_dataset(args)

    for X, Y, mask in te_loader:
        # print(X.shape, Y.shape)
        print(X.shape, Y.shape, mask.shape, X.min(), X.max(), Y.min(), Y.max(), mask.sum())