import os
import torch
import numpy as np
from torchvision import transforms


def load_file(files):
    """
    files (lst) contains one or more than one file path.
    """
    data = np.load(files[0])

    if len(files) > 1:
        res = [data]
        for file in files[1:]:
            res.append(np.load(file))
        return np.dstack(res)
    return data


def pad_img(img, pad_size=20, pad_z=0, pad_method='reflect'):
    if img.ndim == 3:
        return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (pad_z, pad_z)), pad_method)
    elif img.ndim == 2:
        return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), pad_method)
    else:
        raise NotImplementedError


def crop_patch(img, x, y, crop_row, crop_col):
    if img.ndim == 3:
        return img[x:x+crop_row, y:y+crop_col, :]
    elif img.ndim == 2:
        return img[x:x+crop_row, y:y+crop_col]
    else:
        raise ValueError


def pad_crop(items, pad_size):
    """
    Pad input arrays X, Y, and the mask matrix Z.
    Then crop the original size out.
    """
    # get original size
    h, w = items[0].shape[0], items[1].shape[1]

    # pad the input arrays
    items = [pad_img(item, pad_size=pad_size) for item in items]

    # crop the padded arrays
    x = np.random.randint(pad_size*2)
    y = np.random.randint(pad_size*2)

    # obtain the final results
    items = [crop_patch(item, x, y, h, w) for item in items]

    return tuple(items)


def random_hflip(items):
    """
    Randomly and horizontally flip the input arrays.

    Parameters:
        items (list) -- list of arrays
    """
    if torch.rand(1) < 0.5:
        return tuple([np.flip(item, axis=1) for item in items])
    return tuple(items)


def random_vflip(items):
    """
    Randomly and vertically flip the input arrays.

    Parameters:
        items (list) -- list of arrays
    """
    if torch.rand(1) < 0.5:
        return tuple([np.flip(item, axis=0) for item in items])
    return tuple(items)


def to_tensor(items):
    """
    Convert list of numpy arrays to tensor
    """

    return tuple([(transforms.ToTensor()(item.copy())).float() for item in items])