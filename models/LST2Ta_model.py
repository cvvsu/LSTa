import os, torch
from torch import nn
from .base import BaseModel
from . import networks

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, out, target, mask):

        # print(out.shape, target.shape, mask.shape)
        # compuate the loss on the pixels with valid values
        diff = torch.abs(out - target) * mask

        l2 = 0.5 * diff ** 2
        l1 = diff - 0.5

        smoothloss = torch.where(diff >= 1, l1, l2)
        # print(torch.sum(mask), torch.sum(~mask), torch.numel(mask))
        return smoothloss.sum() / torch.sum(mask)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, target, mask):
        
        # compuate the loss on the pixels with valid values
        diff = torch.abs(out - target) * mask
        loss = torch.sum(diff ** 2)

        return loss / torch.sum(mask)


class LST2Ta(BaseModel):

    def __init__(self, args):
        BaseModel.__init__(self, args)

        self.net = networks.get_network(args)
        self.net.to(self.device)
        
        self.criterion = SmoothL1Loss().to(self.device)

        self.setup()

    def set_input(self, input):
        self.X = input[0].to(self.device, non_blocking=True)
        self.y = input[1].to(self.device, non_blocking=True)
        self.mask = input[2].to(self.device, non_blocking=True)
    
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss = self.criterion(self.out, self.y, self.mask)
        self.loss.backward()
        self.optimizer.step()

    def update_test_loss(self):
        return self.criterion(self.out, self.y, self.mask)

    @torch.no_grad()
    def test(self, te_loader):
        if self.args.isTest:
            self.load_network()
            self.net.eval()

            # get results
            cols = ['mean', 'pred_mean', 'min', 'pred_min', 'max', 'pred_max']
            df = pd.DataFrame(columns=cols)
            for idx, (X, Y, mask) in enumerate(te_loader):
                X = X.to(self.device)
                pred = self.net(X)
                pred = pred.detach().cpu().numpy() * 50.0
                gt = Y.detach().cpu().numpy() * 50.0
                mask = mask.detach().cpu().numpy().astype(bool)

                y_pred_mean = pred[:, 0::3, :, :].squeeze()[mask[:, 0::3, :, :].squeeze()].reshape(-1, 1)
                y_pred_min = pred[:, 1::3, :, :].squeeze()[mask[:, 1::3, :, :].squeeze()].reshape(-1, 1)
                y_pred_max = pred[:, 2::3, :, :].squeeze()[mask[:, 2::3, :, :].squeeze()].reshape(-1, 1)

                y_gt_mean = gt[:, 0::3, :, :].squeeze()[mask[:, 0::3, :, :].squeeze()].reshape(-1, 1)
                y_gt_min = gt[:, 1::3, :, :].squeeze()[mask[:, 1::3, :, :].squeeze()].reshape(-1, 1)
                y_gt_max = gt[:, 2::3, :, :].squeeze()[mask[:, 2::3, :, :].squeeze()].reshape(-1, 1)

                df_ = pd.DataFrame(np.hstack([y_gt_mean, y_pred_mean, y_gt_min, y_pred_min, y_gt_max, y_pred_max]), columns=cols)
                df = pd.concat([df, df_])

            # print(df.shape)
            for item in ['mean', 'min', 'max']:
                y_gt = df[item]
                y_pred = df['pred_'+item]
                logger.info(f'R2 for {item}: {R2(y_gt, y_pred):.4f}')
                logger.info(f'MAE for {item}: {MAE(y_gt, y_pred):.4f}')
                logger.info(f'RMSE for {item}: {MSE(y_gt, y_pred, squared=False):.4f}')



                

                

               
            
            # get the prediction results
            # preds = torch.cat(preds, dim=0).squeeze().numpy()
            # gts = torch.cat(gts, dim=0).squeeze().numpy()
            # mask = torch.cat(masks, dim=0).squeeze().numpy()
            # print(preds.shape, gts.shape, mask.shape)

            # pred_mean, gt_mean = preds[:, 0::3, :, :].squeeze(), gts[:, 0::3, :, :].squeeze()
            # pred_min, gt_min = preds[:, 1::3, :, :].squeeze(), gts[:, 1::3, :, :].squeeze()
            # pred_max, gt_max = preds[:, 2::3, :, :].squeeze(), gts[:, 2::3, :, :].squeeze()
            # mask = mask.astype(bool)[0]
            # # print(mask.shape)
            # mask_mean, mask_min, mask_max = mask[0, :, :], mask[1, :, :], mask[2, :, :]

            # # print(mask.shape, pred_mean.shape)
            # y_pred_mean = pred_mean.transpose(1, 2, 0)[mask_mean].reshape(-1, 1)
            # y_pred_min = pred_min.transpose(1, 2, 0)[mask_min].reshape(-1, 1)
            # y_pred_max = pred_max.transpose(1, 2, 0)[mask_max].reshape(-1, 1)

            # y_gt_mean = gt_mean.transpose(1, 2, 0)[mask_mean].reshape(-1, 1)
            # y_gt_min = gt_min.transpose(1, 2, 0)[mask_min].reshape(-1, 1)
            # y_gt_max = gt_max.transpose(1, 2, 0)[mask_max].reshape(-1, 1)

            # # print(y_pred_mean.shape, y_gt_mean.shape, y_pred_min.shape, y_gt_min.shape, y_pred_max.shape, y_gt_max.shape)

            # cols = ['mean', 'pred_mean', 'min', 'pred_min', 'max', 'pred_max']
            # df = pd.DataFrame(np.hstack([y_gt_mean, y_pred_mean, y_gt_min, y_pred_min, y_gt_max, y_pred_max]), columns=cols)
            # df.to_csv(os.path.join(self.savedir, 'test_results.csv'), index=False)

            # for item, y_gt, y_pred in zip(['mean', 'min', 'max'], [y_gt_mean, y_gt_min, y_gt_max], [y_pred_mean, y_pred_min, y_pred_max]):
            #     logger.info(f'R2 for {item}: {R2(y_gt, y_pred):.4f}')
            #     logger.info(f'MAE for {item}: {MAE(y_gt, y_pred)*50}')
            #     logger.info(f'MSE for {item}: {MSE(y_gt, y_pred, squared=False)*50}')
            

def build_model(args):
    return LST2Ta(args)
           
