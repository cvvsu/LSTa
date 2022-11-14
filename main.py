from datasets import get_dataset
from models import get_model
from options import get_args
from loguru import logger
import torch.distributed as dist
import os, argparse
from utils import mkdirs, get_sys_info


if __name__=='__main__':
    
    args, msg = get_args()
    savedir = os.path.join(args.ckptdir, args.name)
    # create the folder on the main process
    if args.dist == 'DDP':
        if int(os.environ['RANK']) == 0:
            mkdirs(savedir)

    logger.add(os.path.join(savedir, f'{args.name}.log'))    
    
    if args.dist == 'DDP':
        if (not args.isTest) and (int(os.environ['RANK']) == 0):
            get_sys_info()
            for line in msg.splitlines():
                logger.info(line)
    else:
        if (not args.isTest):
            get_sys_info()
            for line in msg.splitlines():
                logger.info(line)

    # initilize the DDP if specified
    if args.dist == 'DDP':
        dist.init_process_group(backend='nccl', init_method='env://')

    tr_loader, val_loader, te_loader = get_dataset(args)
    model = get_model(args)

    # training
    model.train(tr_loader, val_loader, args.epochs)

    # test
    model.test(te_loader)

