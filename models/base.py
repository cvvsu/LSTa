import os, time
import numpy as np

import torch
from torch.nn import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler

from abc import ABC, abstractmethod
from loguru import logger

from utils import mkdirs

class BaseModel(ABC):
    """
    Base model for other models.
    Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py

    To create a submodel, you need to implement the abstract methods.
    """
    def __init__(self, args):
        """
        Initialize the model.
        
        """

        # initialize the model
        self.args = args
        self.isTrain = not args.isTest
        
        if self.args.dist == 'DDP':

            self.world_size = torch.distributed.get_world_size()  # how many nodes (16)
            self.rank = torch.distributed.get_rank()              # the current node (0, 1, 2, ..., 13, 14, 15)
            # self.local_rank = torch.distributed.local_rank()      # local node (0-4, 0-4, 0-4,  0-4)
            self.local_rank = args.local_rank
            
            if self.rank == 0:  # the main process
                logger.info(f'The world size is {self.world_size}, the rank number is {self.rank}, the local rank number is {self.local_rank}')
            
            self.device = torch.device(f'cuda:{self.local_rank}')

        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.savedir = os.path.join(args.ckptdir, args.name)

        if self.args.dist == 'DDP':
            self.set_random_state(self.args.seed + self.rank)  # use different seed for different processses to make sure that each process has its own dataset samples
        else:
            self.set_random_state(self.args.seed)

        self.net = None

    def set_random_state(self, seed):        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup(self):
        if self.isTrain:

            if self.args.resume:
                self.load_network(self.args.load_epoch)
            else:
                self.init_network()

            if self.args.dist == 'DP':
                # use all gpu devices
                self.net = torch.nn.DataParallel(self.net)
            elif self.args.dist == 'DDP' and torch.cuda.device_count() > 1:
                self.net = DDP(self.net, device_ids=[self.local_rank], output_device=self.local_rank)
            
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
            if self.args.nowarmup:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs, eta_min=1e-7)
            else:
                self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.args.epochs, lr_min=1e-7, warmup_t=5, warmup_lr_init=1e-7, warmup_prefix=True)
        
        self.print_network()

    def set_input(self, input):
        self.X = input[0].to(self.device, non_blocking=True)
        self.y = input[1].to(self.device, non_blocking=True)
        # print(self.X.shape, self.y.shape)
    
    def forward(self):
        self.out = self.net(self.X)
        # print(self.out, self.y)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss = self.criterion(self.out, self.y)
        self.loss.backward()
        self.optimizer.step()

    def update_test_loss(self):
        return self.criterion(self.out, self.y)
    
    def update_lr(self, epoch):
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.args.nowarmup:
            self.scheduler.step()
        else:
            self.scheduler.step(epoch)
        new_lr = self.optimizer.param_groups[0]['lr']
        return old_lr, new_lr
    
    def train_one_epoch(self, loader):
        tic = tic_data = time.time()
        loss, acc = 0.0, 0.0
        data_times = 0.0

        self.net.train()

        for data in loader:
            data_times += time.time() - tic_data
            self.set_input(data)
            self.forward()
            self.optimize_parameters()
            loss += self.loss.item()

            # acc is only used for classification models
            if self.args.metric == 'accuracy':
                acc += (self.out.argmax(dim=-1) == self.y).float().mean()

            tic_data = time.time()
        return loss / len(loader), acc / len(loader), time.time() - tic, data_times / len(loader)
    
    @torch.no_grad()
    def test_one_epoch(self, loader):
        tic = tic_data = time.time()
        if self.args.dist == 'DDP':
            loss, acc = torch.zeros((1,), device=self.device), torch.zeros((1,), device=self.device)
        else:
            loss, acc = 0.0, 0.0
        data_times = 0.0 
        
        self.net.eval()
        for data in loader:
            data_times += time.time() - tic_data
            self.set_input(data)
            self.forward()
            loss += self.update_test_loss().item()

            # acc is only used for classification models
            if self.args.metric == 'accuracy':
                acc += (self.out.argmax(dim=-1) == self.y).float().mean()

            tic_data = time.time()
        
        # reduce the results to the main process
        if self.args.dist == 'DDP':
            torch.distributed.reduce(loss, 0)
            torch.distributed.reduce(acc, 0)
            loss = loss.item()
            acc = acc.item()
        return loss / len(loader), acc / len(loader), time.time() - tic, data_times / len(loader)
    
    def init_network(self):
        """
        Initialize the networks.
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, 0.02)

                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        self.net.apply(init_func)
    
    def load_network(self, epoch='best'):
        """
        Load the network.
        Load the parameters before sending the model to DP or DDP
        """
        load_path = os.path.join(self.savedir, f'net_{epoch}.pth')
        logger.info(f'Loading the pretrained model from {load_path}.')
       
        state_dict = torch.load(load_path, map_location=self.device)        
        self.net.load_state_dict(state_dict)
    
    def save_network(self, epoch='best'):
        """
        Save the network.
        """
        save_path = os.path.join(self.savedir, f'net_{epoch}.pth')
        if self.args.dist == 'DP':
            torch.save(self.net.module.state_dict(), save_path)
        elif self.args.dist == 'DDP' and self.rank == 0:  # save on the main process
            torch.save(self.net.module.state_dict(), save_path)
        else:
            torch.save(self.net.state_dict(), save_path)

    def print_network_(self):
        logger.info('-'*40)
        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        if self.args.verbose:
            logger.info(self.net)        
        logger.info(f'Total number of trainable parameters is {num_params / 1e6 :.4f}M.')
        logger.info('-'*40)

    def print_network(self):
        if self.args.dist == 'DDP':
            if self.rank == 0:
                self.print_network_()
        else:
            self.print_network_()

    def log_info(self, info):
        """
        log information.
        """
        if self.args.dist == 'DDP':
            if self.rank == 0:
                logger.info(info)
        else:
            logger.info(info)        

    def train(self, tr_loader, val_loader, epochs):
        if self.isTrain:
            best_loss = np.inf
            best_acc = 0
            early_stop_cnt = 0
            writer = SummaryWriter(log_dir=self.savedir)  # writing the loss for visualization in tensorboard

            for epoch in range(1, epochs+1):

                # shuffle the train and validation loaders
                if self.args.dist == 'DDP':
                    tr_loader.sampler.set_epoch(epoch)
                    val_loader.sampler.set_epoch(epoch)

                tr_loss, tr_acc, tr_time, tr_data = self.train_one_epoch(tr_loader)
                val_loss, val_acc, val_time, val_data = self.test_one_epoch(val_loader)
                writer.add_scalar('Loss/train', tr_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Acc/train', tr_acc, epoch)
                writer.add_scalar('Acc/val', val_acc, epoch)
                writer.add_scalar('Time/train', tr_time, epoch)
                writer.add_scalar('Time/val', val_time, epoch)

                if (self.args.metric == 'loss')  and (val_loss <= best_loss):
                    best_loss = val_loss
                    self.save_network('best')
                    early_stop_cnt = 0
                elif (self.args.metric == 'accuracy') and (val_acc >= best_acc):
                    best_acc = val_acc
                    self.save_network('best')
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                
                if early_stop_cnt >= self.args.early_stop:
                    logger.info(f'Early stop at epoch {epoch}.')
                    return
                
                old_lr, new_lr = self.update_lr(epoch)

                msg = f'Epoch {epoch}, lr: [{old_lr:.6f}->{new_lr:.6f}], loss [{tr_loss:.4f}<->{val_loss:.4f}], acc [{tr_acc*100:.2f}%<->{val_acc*100:.2f}%], time [{tr_time:.2f}<->{val_time:.2f}]s, data [{tr_data:.6f}<->{val_data:.6f}]'
                self.log_info(msg)

            self.log_info('-----Training done-----')

        


            
