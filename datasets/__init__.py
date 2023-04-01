import sys
sys.path.append("..")
from .q_o_qo import QueryOfferQueryxOfferDataset, qoNpzfilelistDataset, qoTextfilelistDataset, qoDataProcessor
from .samplers import DistributedSamplerViaLocallyShuffle
from .helper import *
from enum import Enum
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import logging
import torch

logger = logging.getLogger(__name__)


class DatasetsEnum(Enum):
    q_o_qo_dataset = "q_o_qo_dataset"
    qo_dataset = "qo_dataset"

def instantiate(config, process_params, mode):
    if config['type'] == DatasetsEnum.q_o_qo_dataset.value:
        dataset = QueryOfferQueryxOfferDataset(querynpz_dir=config['querynpz_dir'],
                                               offernpz_dir=config["offernpz_dir"],
                                               train_data_num=config["train_data_num"])
        sampler = DistributedSampler(dataset=dataset,
                                     num_replicas=process_params['world_size'],
                                     rank=process_params['local_rank'],
                                     shuffle=config['shuffle'],
                                     seed=config['seed'],
                                     drop_last=config['drop_last'])
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                num_workers=config['num_cpu_workers'],
                                batch_size=config['batch_size'],
                                shuffle=config['shuffle'],
                                drop_last=config['drop_last'],
                                collate_fn=dataset.collate,
                                pin_memory=config['use_pin_memory'])
        return dataloader
    elif config['type'] == DatasetsEnum.qo_dataset.value:
        return qoDataLoaderWrapper(process_params, config, mode=mode)
    else:
        raise Exception("unknown dataset type")


class DataLoaderWrapper:
    '''
    this is just a skeletal structure of a dataloader
    '''
    def __init__(self):
        return
    
    def dataset_init(self):
        return
    
    def sampler_init(self):
        return
    
    def on_epoch_end(self):
        return
    
    def __iter__(self):
        return
    
    def __len__(self):
        return
    
    def state_dict(self):
        return
    
    def load_state_dict(Self):
        return


class qoDataLoaderWrapper(DataLoaderWrapper):
    def __init__(self, process_params, config, mode):
        super(qoDataLoaderWrapper).__init__()
        logger.info(f'in qoDataLoaderWrapper with mode {mode}')
        self.local_rank = process_params['local_rank']
        self.device = process_params['device']
        self.world_size = process_params['world_size']
        self.mode = mode
        self.config = config
        self.epoch = 0
        self.step = 0 # this is always in epoch step
        self.dataset, self.total_records_in_files, self.file_to_len_mapping = self.dataset_init()
        self.sampler, self.dl_batch_size = self.sampler_init()
        # remember all the standard samplers length is = len(dataset) 
        # a dataloader length is len(sampler)
        self.dl = DataLoader(dataset=self.dataset,
                            sampler=self.sampler,
                            num_workers=self.config['loader'].get('num_cpu_workers', 2),
                            batch_size=self.dl_batch_size,
                            shuffle=self.config['loader'].get('shuffle', False),
                            drop_last=self.config.get('drop_last', True),
                            collate_fn=self.dataset.collate,
                            pin_memory=config['loader'].get('use_pin_memory', False))
    
    def dataset_init(self):
        datainit = qoDataProcessor(load_via=self.config['load_via'],
                        file_root=self.config['root'],
                        batch_size=self.config['batch_size'],
                        drop_last=self.config.get('drop_last', True),
                        shuffle_buffer=self.config.get('shuffle_buffer', 0),
                        mode=self.mode)
        data = datainit.load()
        file_to_len_mapping = parse_files_len(root_dir=self.config['root'], 
                                              file_name=self.config.get('file_to_len_mapping', None))
        total_records_in_files = int(sum([file_to_len_mapping[k] for k in file_to_len_mapping.keys()]))
        logger.info(f'total_records : {total_records_in_files}')
        if self.config['load_via'] == "npzfilelist":
            return qoNpzfilelistDataset(fileinfo=data['data'], 
                                        batch_offsets=data['labels'],
                                        input_schema=[i.split(':') for i in self.config['input_schema'].split(',')],
                                        use_segment_id = self.config['use_segment_id'],
                                        useful_cols = self.config['useful_cols'].split(',')), total_records_in_files, file_to_len_mapping
        elif self.config['load_via'] in ["textfilelist", "textfile"]:
            return qoTextfilelistDataset(fileinfo=data['data'], 
                                         batch_offsets=data['labels'],
                                         max_n_seqq=self.config['max_seq_len_q'],
                                         max_n_seqdoc=self.config['max_seq_len_doc'],
                                         input_schema=[i.split(':') for i in self.config['input_schema'].split(',')],
                                         useful_cols=self.config['useful_cols'].split(','),
                                         use_segment_id = self.config['use_segment_id']), total_records_in_files, file_to_len_mapping
        else:
            logger.exception("unknown load_via in qoDataLoaderWrapper")
            sys.exit(1)

    def sampler_init(self):
        if self.world_size > 1:
            if self.config['sampler']['name'] == "DistributedSamplerViaLocallyShuffle" and self.config.get('shuffle_buffer', 0) > 0:
                # call find_checkpoint function when restoring
                # this is used in npzfilelist mode so each dataset.getitem is one batch
                return DistributedSamplerViaLocallyShuffle(dataset=self.dataset, 
                                                            reader=file_reader,
                                                            num_replicas=self.world_size,
                                                            rank=self.local_rank,
                                                            shuffle_buffer=self.config.get('shuffle_buffer', self.config['batch_size']*2),
                                                            total_size=self.total_records_in_files,
                                                            batch_size=self.config['batch_size'],
                                                            file_buffer=self.config.get('file_buffer', 10),
                                                            files_len=self.file_to_len_mapping), 1
            else:
                # store seed in state_dict for restoration
                # this is used in textfilelist mode so each dataset.getitem is one batch
                return DistributedSampler(dataset=self.dataset,
                                            num_replicas=self.world_size,
                                            rank=self.local_rank,
                                            shuffle=self.config['sampler']['shuffle'],
                                            drop_last=self.config.get('drop_last', True)), 1
        else:
            return None, 1

    def on_epoch_end(self):
        self.epoch += 1
        self.step = 0
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self.epoch)
        if hasattr(self.sampler, 'seed'):
            self.sampler.seed = self.epoch
        if hasattr(self.sampler, 'generator'):
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            self.generator = generator

    def __iter__(self):
        for batch in self.dl:
            self.step += 1
            yield batch
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.dl)

    def state_dict(self):
        return {'epoch': self.epoch, 'step': self.step}
    
    def load_state_dict(self, sd):
        self.epoch = sd['epoch']
        if self.config['sampler']['name'] == "DistributedSamplerViaLocallyShuffle" and self.config.get('shuffle_buffer', 0) > 0:
            self.sampler.find_ckpt_position(sd['step'])


