import os, sys
import argparse
import torch.multiprocessing as mp
import logging
import torch.distributed as dist
import torch
import numpy as np
import random
import torch.nn as nn
import yaml

import execution
import datasets
import models

logger = logging.getLogger(__name__)


def init_logger(node_rank, global_rank, local_rank, log_level, log_dir, stdout_only=False):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(filename=os.path.join(log_dir, f'logs_nr-{node_rank}_lr-{local_rank}_gr-{global_rank}.txt'))
        handlers.append(file_handler)
    logging.basicConfig(
            format=f"%(asctime)s - %(levelname)s - nr_{node_rank} - lr_{local_rank} - gr_{global_rank} - %(name)s(%(filename)s:%(lineno)d) - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=log_level,
            handlers=handlers
        ),
    return logger


def process_init(node_rank, global_rank, local_rank, world_size, device, log_dir, log_level):
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=global_rank
        )
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    init_logger(node_rank, global_rank, local_rank, log_level, log_dir)


def run(local_rank, config, devices, args):
    logger.info('testing the logger')
    global_rank = args.node_rank * args.num_gpus_per_node + local_rank
    device = devices[local_rank]
    process_init(args.node_rank, global_rank, local_rank, config['world_size'], device, args.log_dir, args.log_level)
    process_params = {'node_rank': args.node_rank,
                      'local_rank': local_rank,
                      'global_rank': global_rank,
                      'world_size': config['world_size'],
                      'device': device}
    print('config : ', config)
    logger.info(f'using device {device}')
    if args.run_mode == 'train':
        model = models.instantiate(config['exec']['model'], process_params)
        #model.print_summary()
        if config['world_size'] > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
        train_dataloader = datasets.instantiate(config['exec']['data']['train'], process_params, mode='train')
        validation_dataloader = datasets.instantiate(config['exec']['data']['validation'], process_params, mode='validation')
        trainer = execution.Train(config=config,
                                  process_params=process_params,
                                  model=model,
                                  train_dataloader=train_dataloader,
                                  validation_dataloader=validation_dataloader)
        trainer.fit()
    elif args.run_mode == 'test':
        model = models.instantiate(config['exec']['model'], process_params)
        if config['world_size'] > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
        test_dataloader = datasets.instantiate(config['exec']['data']['test'], process_params)
    else:
        logger.exception("unknown run_mode, exiting...")


def parseargs(parser):
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--num_nodes', default=1, required=False, type=int)
    parser.add_argument('--num_gpus_per_node', default=1, required=False, type=int)
    parser.add_argument('--node_rank', default=0, required=False, type=int)
    parser.add_argument('--devices', default='cuda:0', required=False, type=str)
    parser.add_argument('--master_address', default='localhost', required=False, type=str)
    parser.add_argument('--master_port', default='44401', required=False, type=str)
    parser.add_argument('--run_mode', default='train', required=True, type=str, help='use train/test/infer')
    parser.add_argument('--log_dir', default='', required=True, type=str)
    parser.add_argument('--log_level', default='INFO', required=False, type=str)
    return parser


def parse_config_from_yaml(configFilePath):
    with open(configFilePath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def overwrite_from_args(config, args):
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)


if __name__ == '__main__':
    '''
    this code works with all the below settings
    1) 1 cpu
    2) 1 gpu
    3) 1 node, n gpus
    4) m nodes, n gpus
    '''
    parser = argparse.ArgumentParser()
    parser = parseargs(parser)
    args, unknown = parser.parse_known_args()
    for arg in unknown:
        print("unknown args passed {unknown}")
    config = parse_config_from_yaml(args.config)
    overwrite_from_args(config, args)
    config['world_size'] = args.num_nodes * args.num_gpus_per_node
    devices = args.devices.split(',')
    if args.num_nodes > 1 or args.num_gpus_per_node > 1:
        print('running a multi gpu execution')
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = args.master_port
        try:
            mp.spawn(run, nprocs=config['world_size'], args=(config, devices, args), join=True)
        except KeyboardInterrupt:
            logger.warn('Interrupted')
            try:
                torch.distributed.destroy_process_group()
            except AssertionError:
                pass
    else:
        print("running a single cpu/gpu execution")
        run(local_rank=0, config=config, devices=devices, args=args)