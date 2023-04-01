import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import logging
import tqdm
import time
from utils import *
import glob
import copy
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Train():
    def __init__(self, config, process_params, 
                model, train_dataloader, validation_dataloader):
        self.config = config
        self.device = process_params['device']
        self.device_id = int(self.device.split(':')[-1])
        self.local_rank = process_params['local_rank']
        self.global_rank = process_params['global_rank']
        self.world_size = process_params['world_size']
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.per_epoch_total_epochs = len(self.train_dataloader)
        self.model = model
        self.optimizer, self.lr_scheduler = self.model.configure_optimizer(self.per_epoch_total_epochs)
        self.epoch = 0
        self.global_step = 0
        self.in_epoch_step = -1
        self.output_dir = self.config['output_dir']
        self.log_dir = self.config['log_dir']
        self.tb_summary_dir = self.output_dir + '/tb_runs/orcas/npz'
        self.profiler_dir = self.tb_summary_dir + '/profiler'
        self.checkpoint_dir = self.output_dir + '/checkpoints'
        os.makedirs(self.profiler_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tb_summary_dir, exist_ok=True)
        self.save_interval = self.config['save_interval']
        self.last_saved = time.time()
        self.tb_summary_writer = SummaryWriter(self.tb_summary_dir)
        if self.config['exec'].get('track_training_dynamics', None):
            self.training_dynamics_fp = open(os.path.join(self.log_dir, 'training_dynamics.txt'),
                                             mode='a',
                                             encoding='utf-8',
                                             buffering=1)

    def fit(self):
        self.preemptive_restore()
        self.load_model_from_checkpoint()
        #self.validate_on_train_start()
        self.model.train()

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        for epoch in range(0, self.config['epochs']):
            self.tb_summary_writer.flush()
            self.epoch = epoch
            self.in_epoch_step = -1
            if self.config['profile']:
                pbar = self.train_dataloader
                prof.start()
            else:
                pbar = tqdm(self.train_dataloader, 
                                desc=f"epoch-{epoch} progress => loss: inf", 
                                ascii=' |', 
                                disable=(self.local_rank != 0 or self.global_step == 0), 
                                initial=self.in_epoch_step+1)
            for self.in_epoch_step, x_batch in enumerate(pbar, start=self.in_epoch_step+1):
                x_batch = self.model.totorchtensor(x_batch)
                self.optimizer.zero_grad()                    
                loss = self.model.train_step(x_batch)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                if self.in_epoch_step%self.config['print_after_steps']==0 and self.local_rank==0:
                    if self.in_epoch_step == 0:
                        self.plot_grad_flow(self.model.named_parameters(), epoch, self.in_epoch_step)
                    if self.global_step==0:
                        with torch.no_grad():
                            self.tb_summary_writer.add_graph(self.model, x_batch)
                            self.model.print_summary()
                    self.save_checkpoint()
                    if self.config['profile']:
                        logger.info(f"epoch-{epoch} => batch-{self.in_epoch_step}/{self.per_epoch_total_epochs} => loss: {loss.item():.5f}")
                    else:
                        pbar.set_description(f"epoch-{epoch} progress => loss: {loss.item():.5f}")
                self.tb_summary_writer.add_scalar("train_loss", loss.item(), self.global_step)
                self.global_step += 1
                if self.config['profile'] and self.epoch==0 and self.in_epoch_step <= (1+1+1)*2:
                    prof.step()
            if self.config['profile']:
                prof.stop()
            else:
                pbar.close()
            if self.local_rank == 0:
                logger.info("epoch " + str(epoch) + " end")
                self.validate_on_epoch_end()
                self.save_checkpoint(force=True)
                if self.world_size > 1:
                    dist.barrier()

    def validate_on_epoch_end(self):
        self.model.eval()
        pbar = tqdm(self.validation_dataloader, 
                            desc=f'validation progress', 
                            ascii=' |', 
                            disable=(self.local_rank != 0))
        output_batch = []
        for batch_idx, x_batch in enumerate(pbar):
            x_batch = self.validation_dataloader.totorchtensor(x_batch, self.device_id)
            output_batch.append(self.model.validate_step(x_batch))
        logger.info(f'validation results at end of epoch :\n{self.model.compute_metrics(output_batch)}')
    
    def validate_on_train_start(self):
        self.model.eval()
        pbar = tqdm(self.validation_dataloader, 
                            desc=f'validation progress', 
                            ascii=' |', 
                            disable=(self.local_rank != 0),
                            colour='green')
        output_batch = []
        for batch_idx, x_batch in enumerate(pbar):
            x_batch = self.validation_dataloader.totorchtensor(x_batch, self.device_id)
            output_batch.append(self.model.validate_step(x_batch))
        logger.info(f'validation results at start of the training :\n{self.model.compute_metrics(output_batch)}')
         
    def save_checkpoint(self, force=False):
        if (not force) and (self.save_interval <= 0):
            return
        time_now = time.time()
        if self.world_size > 1:
            time_now = all_sync(time_now, self.world_size, 0)
        if force or (time_now - self.last_saved > self.save_interval):
            state_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'data': all_gather(self.train_dataloader.state_dict(), self.world_size),
                'rng': all_gather({'torch': torch.get_rng_state()}, self.world_size),
                'iter': all_gather({'epoch': self.epoch, 
                                    'in_epoch_step': self.in_epoch_step,
                                    'global_step': self.global_step}, self.world_size)
            }
            ckpt_path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.epoch}_{self.in_epoch_step}.pt')
            if self.local_rank == 0:
                torch.save(state_dict, ckpt_path)
            self.last_saved = time_now
            logger.info(f'saved checkpoint at {ckpt_path}')
            return ckpt_path
        else:
            return None
    
    def preemptive_restore(self):
        load_from_ckpt = self.config['restore_from_checkpoint_path']
        if self.config['restore_from_checkpoint_path'] is None:
            if (not os.path.exists(self.checkpoint_dir)) or len(os.listdir(self.checkpoint_dir))==0:
                return
            list_of_ckpts = glob.glob(self.checkpoint_dir+'/*')
            load_from_ckpt = max(list_of_ckpts, key=os.path.getctime)
        logger.info("restoring from ckpt {load_from_ckpt}")
        ckpt = torch.load(load_from_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.train_dataloader.load_state_dict(ckpt['data'][self.global_rank])
        torch.set_rng_state(ckpt['rng'][self.global_rank]['torch'].cpu())
        self.epoch = ckpt['iter'][self.global_rank]['epoch']
        self.in_epoch_step = ckpt['iter'][self.global_rank]['in_epoch_step']
        self.global_step = ckpt['iter'][self.global_rank]['global_step']

    def load_model_from_checkpoint(self):
        if self.config['load_model_from_ckpt'] is None:
            return
        logger.warn("model is being loaded from checkpoint, this will overwrite the restore_from_checkpoint!!!")
        ckpt = torch.load(self.config['load_model_from_ckpt'], map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
    
    def verify_model_same_across_devices(self):
        if self.world_size <= 1:
            return
        sds = all_gather({k: v.cpu() for k, v in self.model.state_dict().items()}, self.world_size)
        if self.rank == 0:
            for k in sds[0].keys():
                for i in range(1, self.world_size):
                    if not torch.all(torch.eq(sds[0][k], sds[1][k])):
                        logger.info(f'model different across workers. key = {k}')
                        return
            logger.info('model same across workers')

    def plot_grad_flow(self, named_parameters, epoch, step):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
        plt.rcParams["figure.figsize"] = (30, 10)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=0, top=1e-4)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average absolute gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-abs-gradient', 'mean-abs-gradient', 'zero-abs-gradient'])
        plot_out_path = os.path.join(self.log_dir, "gradient_flow_plots")
        os.makedirs(plot_out_path, exist_ok=True)
        plt.savefig(os.path.join(plot_out_path, 'grad_flow_e' + str(epoch) + "_s" + str(step) + '.png'), bbox_inches='tight')

    def write_training_dynamics_to_file(self, step, training_dynamics):
        line = str(self.epoch) + '\t' + str(step) + '\t' + '\t'.join(
            [key + ":" + str(training_dynamics[key]) for key in training_dynamics]) + '\n'
        self.training_dynamics_fp.write(line)
