import torch.nn.functional as F
from torch import nn

def loss_factory(lossName):
    if lossName == "nll":
        def nll_loss(output, target):
            return F.nll_loss(output, target)
        return nll_loss
    elif lossName == 'BCE':
        return nn.BCEWithLogitsLoss()
        # def BCE_loss(output, target):
        #     return F.binary_cross_entropy(output,target)
        # return BCE_loss
    elif lossName == 'MSE':
        def MSE_loss(output, target):
            return F.mse_loss(output,target)
        return MSE_loss
    elif lossName == 'l1':
        return nn.L1Loss()