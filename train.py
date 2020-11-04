# from multiprocessing import freeze_support
import argparse
import collections
import torch
import numpy as np
import moGAN_git.data_loader.data_loaders as module_data
import moGAN_git.model.loss as module_loss
import moGAN_git.model.metric as module_metric
import moGAN_git.model.model as module_arch
from moGAN_git.parse_config import ConfigParser
from moGAN_git.trainer import Trainer

# if __name__ == '__main__':
#     freeze_support()

# fix random seeds for reproducibility
from moGAN_git.trainer.adv_trainer import AdvTrainer

# import wandb
# wandb.init(project="bvh")

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    gen_model = config.init_obj('Generator',module_arch)
    disc_model = config.init_obj('Discriminator',module_arch)

    # wandb.watch(gen_model)
    # wandb.watch(disc_model)
    logger.info(gen_model)
    logger.info(disc_model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    loss_name = config['loss']
    criterion = module_loss.loss_factory(loss_name)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, gen_model.parameters())
    gen_optimizer = config.init_obj('gen_optimizer', torch.optim, gen_trainable_params )
    disc_trainable_params = filter(lambda p: p.requires_grad, disc_model.parameters())
    disc_optimizer = config.init_obj('disc_optimizer', torch.optim, disc_trainable_params)

    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, gen_optimizer)
    lr_scheduler = None
    # def __init__(self, gen_model,disc_model, criterion, metric_ftns, gen_optimizer,disc_optimizer, config, data_loader,z_dim=10,
    #              test_generator = False, display_step=None, valid_data_loader=None, lr_scheduler=None, len_epoch=None):

    trainer = AdvTrainer(gen_model,disc_model, criterion, metrics, gen_optimizer,disc_optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='moGAN trainer')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
