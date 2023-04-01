import sys
sys.path.append("..")
from .helper import *
from torch.utils.data import Dataset
import os
import numpy as np
from zipfile import ZipFile
import torch.distributed as dist
from transformers import AutoTokenizer
import math
import logging
from time import time

logger = logging.getLogger(__name__)

class qoDataProcessor:

    def __init__(self,
                 load_via,
                 file_root,
                 dataset_path = '',
                 batch_size = 1024, # use this when you read with text
                 drop_last = True, # use this when you read with text
                 skip_files = None,
                 shuffle_buffer = 0,
                 multi_test_set = False,
                 mode = 'train'):
        self.dataset_path = dataset_path
        self.file_root = file_root
        self.load_via = load_via
        self.mode = mode
        if 'list' in self.load_via and os.path.isdir(self.file_root):
            self.file_fullpath = []
            self.file_name = []
            for name in os.listdir(self.file_root):
                if skip_files is not None and name in skip_files:
                    logger.info('Drop ' + name + ' from file list')
                    continue
                self.file_fullpath.append(os.path.join(self.file_root, name))
                self.file_name.append(name)
        else:
            self.file_fullpath = [self.file_root]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_buffer = shuffle_buffer
        self.multi_test_set = multi_test_set
        self.dict = dict()

    def load(self):
        data = None
        labels = None
        
        if self.load_via == 'npzfilelist':
            if self.mode == 'train':
                logger.info('process train file (npzfilelist)')
                print(self.file_fullpath)
                filelist = self.file_fullpath
                data = []
                labels = []
                j = 0
                for i in range(len(filelist)):
                    f = filelist[i]
                    if not f.endswith('npz'):
                        continue
                    data.append(self.dataset_path + f)
                    if self.shuffle_buffer == 0:
                        tmp_data = np.load(self.dataset_path + f)
                        examples_num = len(tmp_data['arr_0'])
                        index = [[j, offset * self.batch_size] for offset in range(examples_num // self.batch_size + 1)]
                        if self.drop_last:
                            index = index[:-1]
                        labels = labels + index
                        j += 1
                    else:
                        labels = None
                self.dict['data'] = [data, self.batch_size]
                self.dict['labels'] = labels
            else: # mode = 'validation' | 'test'
                logger.info('process val file (npzfilelist)')
                filelist = self.file_fullpath
                data = []
                labels = []
                j = 0
                for i in range(len(filelist)):
                    f = filelist[i]
                    if not f.endswith('npz'):
                        continue
                    tmp_data = np.load(self.dataset_path + f)
                    examples_num = len(tmp_data['arr_0'])
                    if not self.multi_test_set:
                        data.append(self.dataset_path + f)
                        index = [[j, offset * self.batch_size] for offset in range(examples_num // self.batch_size + 1)]
                        labels = labels + index
                    else:
                        data.append([self.dataset_path + f])
                        index = [[j, offset * self.batch_size] for offset in range(examples_num // self.batch_size + 1)]
                        labels.append(index)
                    j += 1
                self.dict['data'] = [data, self.batch_size]
                self.dict['labels'] = labels

        if self.load_via == 'textfile':
            logger.info('process file (textfile)')
            data = self.dataset_path + self.file_fullpath[0]
            labels = parse_textfile(self.batch_size, self.dataset_path + self.file_fullpath[0], self.drop_last, return_last=False)
            self.dict['data'] = [data, self.batch_size]
            self.dict['labels'] = labels
        
        if self.load_via == 'textfilelist':
            logger.info('process file (textfilelist)')
            filelist = self.file_fullpath
            data = []
            labels = []
            for i in range(len(filelist)):
                f = filelist[i]
                if os.path.isdir(self.dataset_path + f):
                    continue
                offsets = parse_textfile(self.batch_size, self.dataset_path + f, self.drop_last, return_last=False)
                if not self.multi_test_set:
                    data.append(self.dataset_path + f)
                    index = [[i, offset] for offset in offsets]
                    labels = labels + index #extends the list
                else:
                    data.append([self.dataset_path + f])
                    index = [[0, offset] for offset in offsets]
                    labels.append(index)
            self.dict['data'] = [data, self.batch_size]
            self.dict['labels'] = labels

        if self.load_via == 'largetextfile':
            logger.info('process file (largetextfile)')
            data = self.dataset_path + self.file_fullpath[0]
            max_idx, max_rows = parse_textfile(self.batch_size, data, self.drop_last, return_last=True)
            n_batches = math.floor(max_rows / self.batch_size)
            scalar = n_batches / 1000
            labels = [0] * 1000
            self.dict['data'] = [data, self.batch_size, str(scalar), max_idx]
            self.dict['labels'] = labels

        return self.dict


class qoTextfilelistDataset(Dataset):
    def __init__(self, fileinfo, batch_offsets, max_n_seqq, max_n_seqdoc,
                 input_schema, useful_cols, use_segment_id):
        self.fileinfo = fileinfo
        self.batch_offsets = batch_offsets
        self.max_n_seqq = max_n_seqq
        self.max_n_seqdoc = max_n_seqdoc
        self.berttokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_type = 'wordpiece'
        self.input_schema = input_schema
        self.useful_cols = useful_cols
        self.use_segment_id = use_segment_id
        self.input_cols = [i[0] for i in self.input_schema]
        self.input_types = [i[1] for i in self.input_schema]

    def __len__(self):
        return len(self.batch_offsets)

    def __getitem__(self, i):
        fid = self.batch_offsets[i][0] # switched to textfile list support
        oid = self.batch_offsets[i][1]
        batch = text_to_tokenids(self.fileinfo[0][fid], oid, self.fileinfo[1],
                                self.berttokenizer, self.input_cols,
                                self.input_types, self.useful_cols,
                                max_n_seq=self.max_n_seqq,
                                max_n_seqdoc=self.max_n_seqdoc,
                                use_segment_id=self.use_segment_id)
        return batch

    def collate(self, batch):
        return batch[0]


class qoNpzfilelistDataset(Dataset):
    def __init__(self, fileinfo, batch_offsets, input_schema, use_segment_id, useful_cols):
        self.fileinfo = fileinfo
        self.past_files_samples = []
        self.files = fileinfo[0]
        self.batch_offsets = batch_offsets
        self.input_schema = input_schema
        self.use_segment_id = use_segment_id
        self.number = 0
        self.debug = False
        self.useful_cols = useful_cols
        self.input_cols = [i[0] for i in self.input_schema]
        self.input_types = [i[1] for i in self.input_schema]

    def __len__(self):
        return len(self.batch_offsets)

    def __getitem__(self, i):
        if isinstance(i, int):
            fid = self.batch_offsets[i][0]
            oid = self.batch_offsets[i][1]
            bs = self.fileinfo[1]
            results = tokenids_from_npz(self.files[fid], oid, 
                                        bs, self.input_cols,
                                        self.input_types, self.useful_cols, -1,
                                        data=None,
                                        use_segment_id=self.use_segment_id)
        else:
            files_data = i[0]
            batch_ids = i[1]
            files_path = i[2]
            if batch_ids is None:
                batch_ids = [None] * len(files_data)
            batch = []
            for j in range(len(files_data)):
                batch.append(tokenids_from_npz(files_path[j], batch_ids[j], 
                                               1, self.input_cols,
                                                self.input_types, self.useful_cols, -1,
                                                data=files_data[j],
                                                use_segment_id=self.use_segment_id))
            results = dict()
            for k in batch[0].keys():
                results[k] = np.concatenate([tmp[k] for tmp in batch], axis=0)
            self.number += 1
        return results

    def reset(self):
        self.past_files_samples = []

    def collate(self, batch):
        return batch[0]


class QueryOfferQueryxOfferDataset(Dataset):

    def __init__(self, queryxoffernpz_dir, querynpz_dir, offernpz_dir, train_data_num,
                 qid_to_file_mapping_file='idToFileMapping.txt', oid_to_file_mapping_file='idToFileMapping.txt',
                 qidxoid_id_to_file_mapping='idToFileMapping.txt'):
        self.qo_files_dir = queryxoffernpz_dir
        self.q_files_dir = querynpz_dir
        self.o_files_dir = offernpz_dir
        self.length = train_data_num
        self.qid_to_file_mapping = self.read_id_to_file_mapping(os.path.join(self.q_files_dir, qid_to_file_mapping_file),
                                                           self.q_files_dir)
        self.oid_to_file_mapping = self.read_id_to_file_mapping(os.path.join(self.o_files_dir, oid_to_file_mapping_file),
                                                           self.o_files_dir)
        self.qidxoid_id_to_file_mapping = self.read_id_to_file_mapping(
            os.path.join(self.qo_files_dir, qidxoid_id_to_file_mapping),
            self.qo_files_dir)
        self.preprocess_data()

    def __len__(self):
        # return total training data row count
        return self.length

    def __getitem__(self, index):
        # using standard distributed sampler which just splits the training data rows to gpus
        # sampler only returns the row ids of queryxoffer
        # index = row_id
        # fetch qid, oid of corresponding row_id from qo_files_dir with the help of qidxoid_id_to_file_mapping
        # fetch query features using qid from q_files_dir with the help of qid_to_file_mapping
        # fetch offer features using oid from o_files_dir with the help of oid_to_file_mapping
        data = self.read_numpyfiles_mmap(self.qidxoid_id_to_file_mapping[index]['npy_files_dir'],
                                    self.qidxoid_id_to_file_mapping[index]['offset'],
                                    mode="qxo")
        q_data = self.read_numpyfiles_mmap(self.qid_to_file_mapping[data[0]]['npy_files_dir'],
                                      self.qid_to_file_mapping[data[0]]['offset'],
                                      mode="q")
        o_data = self.read_numpyfiles_mmap(self.oid_to_file_mapping[data[1]]['npy_files_dir'],
                                      self.oid_to_file_mapping[data[1]]['offset'],
                                      mode="o")

        return {"q_data": q_data, "o_data": o_data}

    def collate(self, batch):
        # todo
        # input is what getitem returns = batch of q_data, o_data
        # stack data per feature for both q, o
        stacked_batch = {'q_input_ids': np.stack([x["q_data"]["input_ids"] for x in batch]),
                   'q_attention_mask': np.stack([x["q_data"]["attention_mask"] for x in batch]),
                   'o_input_ids': np.stack([x["o_data"]["input_ids"] for x in batch]),
                   'o_attention_mask': np.stack([x["o_data"]["attention_mask"] for x in batch])}
        return stacked_batch
    
    @staticmethod
    def read_id_to_file_mapping(file_path, file_dir):
        id_to_file_mapping = dict()
        with open(file_path, mode='r', encoding='utf-8') as fp:
            lines = fp.readlines()
        for line in lines:
            id, npy_files_dir, offset = line.strip().split('\t')
            id = int(id)
            id_to_file_mapping[id]['npy_files_dir'] = os.path.join(file_dir, npy_files_dir)
            id_to_file_mapping[id]['offset'] = int(offset)
        return id_to_file_mapping

    def preprocess_data(self):
        # todo
        return None

    @staticmethod
    def read_numpyfiles_mmap(input_dir, offset, mode):
        if mode == "qxo":
            return np.load(os.path.join(input_dir, 'qid_oid.npy'), mmap_mode='r')[offset]
        elif mode == "q":
            input_ids = np.load(os.path.join(input_dir, 'input_ids.npy'), mmap_mode='r')[offset]
            attention_mask = np.load(os.path.join(input_dir, 'attention_mask.npy'), mmap_mode='r')[offset]
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif mode == "o":
            input_ids = np.load(os.path.join(input_dir, 'input_ids.npy'), mmap_mode='r')[offset]
            attention_mask = np.load(os.path.join(input_dir, 'attention_mask.npy'), mmap_mode='r')[offset]
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            raise Exception("unknow mode from read_numpyfiles_mmap")
