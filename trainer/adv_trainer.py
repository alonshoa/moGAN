import torch
import wandb
from numpy import inf
from tqdm import tqdm

from moGAN_git.base import BaseTrainer
from moGAN_git.logger import TensorboardWriter
from moGAN_git.utils import inf_loop
from moGAN_git.utils.gan_utils import get_disc_loss, get_gen_loss, get_noise, show_tensor_images


class AdvTrainer:
    def __init__(self, gen_model,disc_model, criterion, metric_ftns, gen_optimizer,disc_optimizer, config, data_loader,z_dim=10,
                 test_generator = False, display_step=500, valid_data_loader=None, lr_scheduler=None, len_epoch=None):

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.z_dim = z_dim
        self.test_generator = test_generator
        self.display_step = display_step
        self.cur_step = 0
        self.cur_step = 0
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        self.gen_loss = False
        self.error = False

        # dataloader
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.gen_model = gen_model.to(self.device)
        self.disc_model = disc_model.to(self.device)

        if len(device_ids) > 1:
            self.gen_model = torch.nn.DataParallel(gen_model, device_ids=device_ids)
            self.disc_model = torch.nn.DataParallel(disc_model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        # for real, _ in self.data_loader:
        for real, _ in tqdm(self.data_loader):
            cur_batch_size = len(real)

            real = real.view(cur_batch_size, -1).to(self.device)
            self.disc_model.train()
            self.disc_optimizer.zero_grad()

            disc_loss = get_disc_loss(self.gen_model, self.disc_model, self.criterion, real, cur_batch_size, self.z_dim, self.device)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            self.disc_optimizer.step()

            # For testing purposes, to keep track of the generator weights
            if self.test_generator:
                old_generator_weights = self.gen_model.gen[0][0].weight.detach().clone()
            self.gen_model.train()
            self.gen_optimizer.zero_grad()
            gen_loss,fake_images = get_gen_loss(self.gen_model, self.disc_model, self.criterion, cur_batch_size, self.z_dim, self.device)
            gen_loss.backward(retain_graph=True)
            self.gen_optimizer.step()

            # For testing purposes, to check that your code changes the generator weights
            if self.test_generator:
                try:
                    assert self.lr > 0.0000002 or (self.gen_model.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(self.gen_model.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            # Keep track of the average discriminator loss
            self.mean_discriminator_loss += disc_loss.item() / self.display_step

            # Keep track of the average generator loss
            self.mean_generator_loss += gen_loss.item() / self.display_step

            ### Visualization code ###
            if self.cur_step % self.display_step == 0 and self.cur_step > 0:
                print(
                    f"Step {self.cur_step}: Generator loss: {self.mean_generator_loss}, discriminator loss: {self.mean_discriminator_loss}")
                # fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
                # fake = self.gen_model(fake_noise)

                # show_tensor_images(fake_images,size=fake_images.shape)
                # show_tensor_images(real,size=real.shape)
                self.mean_generator_loss = 0
                self.mean_discriminator_loss = 0

            if self.cur_step % self.save_period == 0 and self.cur_step > 0:
                self._save_checkpoint(epoch)
            self.cur_step += 1

        # return {"gen_loss":gen_loss,"disc_loss":disc_loss,"fake_image":fake_images[0]}
        return {"": "", "gen_loss": gen_loss, "disc_loss": disc_loss}

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            print()
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
                #tensorbord
                # self.writer.add_scalar(key, value)
            # wandb
            # wandb.log(log)

            # evaluate model performance according to configu
            # red metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        self.save_model(self.gen_model,self.gen_optimizer, epoch, save_best)
        self.save_model(self.disc_model,self.disc_optimizer, epoch, save_best)

    def save_model(self, model,optimizer, epoch, save_best):
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        filename = f"{self.checkpoint_dir}/checkpoint-epoch{epoch}_{arch}.pth"
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        raise NotImplementedError

        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

