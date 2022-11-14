import argparse

def get_message(parser, args):
    r"""
    References:
        1.https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def get_args():
    parser = argparse.ArgumentParser('LSTa')

    # basic
    parser.add_argument('--ckptdir', default='checkpoints', type=str, help='folder stores the saved models and results')
    parser.add_argument('--name', type=str, default='experiment', help='name for current experiment')
    parser.add_argument('--seed', type=int, default=233, help='random seed for re-producing')

    # datasets
    parser.add_argument('--input_folders', nargs='+', type=str, default=['datasets/Ta'], help='input folder')
    parser.add_argument('--output_folders', nargs='+', type=str, default=['datasets/npy/LSTD'], help='output folder')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio from the raw training set')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pad_size', type=int, default=20, help='padding size for data augmentation')
    parser.add_argument('--mask_path', type=str, default='datasets/Ta_mask', help='masks used to calculated losses. LST2Ta')
    # parser.add_argument('--rows', type=int, default=1143, help='number of rows')
    # parser.add_argument('--cols', type=int, default=1343, help='number of columns')
    parser.add_argument('--station_loc', type=str, default='datasets/station_loc.csv', help='rows and cols for all stations')

    # network parameters
    parser.add_argument('--network_name', type=str, default='unet', help='unet network')
    parser.add_argument('--conv_type', type=str, default='sconv', help='single, double, se [sconv | dconv | seconv]')
    parser.add_argument('--input_nc', type=int, default=3, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--num_layers', type=int, default=8, help='number of unet levels. For large-size images, please increase the number. Otherwise, decrease the number.')    
    parser.add_argument('--num_filters', type=int, default=64, help='number of filters for the input layer')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta for Adam optimizer')

    # model parameters
    parser.add_argument('--model_name', type=str, default='Ta2LST', help='model name')    

    # training
    parser.add_argument('--batch_size', type=int, default=16, help='mini batch size')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--early_stop', type=int, default=400, help='stop training if model has no improvement for 400 epochs')
    parser.add_argument('--resume', action= 'store_true', help='continue training')
    parser.add_argument('--load_epoch', default='best', help='load the best epoch')
    parser.add_argument('--verbose', action= 'store_true', help='print more information if true')
    parser.add_argument('--dropout', type=float, default=0, help='probability of dropout')
    parser.add_argument('--metric', type=str, default='loss', help='[loss | accuracy]')
    parser.add_argument('--nowarmup', action='store_true', help='will use the warmup for lr if not specified')

    # test
    parser.add_argument('--isTest', action= 'store_true', help='test phase if true')
    parser.add_argument('--save_npy', action='store_true', help='store the predicted files to local device')

    # distributed training
    # NOTE that we do not fully test DDP and PyTorch provides a new version DDP: https://pytorch.org/docs/stable/elastic/run.html
    parser.add_argument('--dist', type=str, default='none', help='multi gpu training [none | DP | DDP]')
    parser.add_argument('--local_rank', type=int, default=-1, help='a necessary argment for DDP training even though no need to set it a value')


    args = parser.parse_args()
    return args, get_message(parser, args)