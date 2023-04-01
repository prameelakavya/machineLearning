import sys
sys.path.append("..")
import random
import numpy as np
import os
import multiprocessing as mp
import mmap
import time
import psutil
import gc
import trace
import pdb
import logging
from tqdm import tqdm
from utils import *
import subprocess
import pickle 
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_textfile(batch_size, file_path, drop_last, return_last=False):
        # n instances
        # n features
        n = batch_size
        ret = []
        cnt = 0
        res = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
        file_len = int(res.stdout.split(' ')[0])
        get_position = True
        if drop_last:
            tmp_file = 'C:\\Users\\thumm\\Documents\\machineLearning\\nlp\\code\\pyTorchProject\\output\\datalist'
            if os.path.exists(tmp_file):
                return pickle.load(open(tmp_file, 'rb'))
        with open(file_path, 'r', encoding = 'utf-8', errors='ignore') as fp:
            logger.info(f'parsing file {file_path} of length {file_len}')
            pbar = tqdm(total=file_len, ascii=' |', colour='green')
            while True:
                position_tmp = fp.tell()
                line = fp.readline().strip('\n\r')
                if line == '':
                    break
                if get_position:
                    ret.append(position_tmp)
                    get_position = False
                if cnt % n == (n-1): # 1023
                    get_position = True
                cnt += 1
                if cnt % 10000 == 0:
                    pbar.update(10000)
            pbar.close()
            if drop_last: # simply drop the last one since you don't know how many rows are in that line, note, only use this in training
                pickle.dump(ret, open(tmp_file, 'wb'))
                return ret[0:-1]
        if return_last:
            return ret[-1], cnt
        return ret # this is not backward compatible, since the old textfile dataloader is not used anywhere, we allow this incompatibility.


def parse_files_len(root_dir, file_name, exists_files=[]):
    if file_name is None: # root_dir is expected to be a path
        res = subprocess.run(['wc', '-l', root_dir], capture_output=True, text=True)
        res = res.stdout.split(' ')
        return {res[1] : int(res[0])}
    dict_path = os.path.join(root_dir, file_name)
    cnt = 0
    ret = dict()
    with open(dict_path, 'r', encoding = 'utf-8', errors='ignore') as fp:
        logger.info('parsing file {f}'.format(f = dict_path))
        while True:
            line = fp.readline().strip('\n\r')
            if line == '':
                break
            line = line.split('\t')
            data_name = line[0]
            l = line[1]
            ret[os.path.join(root_dir, data_name)] = int(l)
            cnt += 1
    logger.info(str(cnt) + ' files in ' + dict_path)
    return ret


def tokenids_from_npz(filename, fpointer, batch_size, input_cols, 
                                    input_types, useful_cols, 
                                    max_idx = -1, data=None, use_segment_id=False): # fpointer == -1, means you can randomize
    if data is None:
        data = np.load(filename, mmap_mode='r')
    cols = dict()
    i = 0
    j = 0
    if fpointer is None:
        for col in useful_cols:
            while input_cols[i] != col:
                j += 1
                if 'str' in input_types[i]:
                    j += 1
                i += 1
            if 'str' in input_types[i] and 'id' not in input_cols[i]:
                cols[col + 'seq'] = data['arr_' + str(j)].astype(np.int32)
                cols[col + 'mask'] = data['arr_' + str(j + 1)].astype(np.int32)
                j += 1
                if use_segment_id:
                    cols[col + 'seg'] = data['arr_' + str(j + 1)].astype(np.int32)
                    j += 1
            elif input_types[i] == 'int':
                cols[col] = data['arr_' + str(j)].astype(np.int32)
            elif input_types[i] == 'str':
                cols[col] = data['arr_' + str(j)]
            else:
                cols[col] = data['arr_' + str(j)].astype(np.float)
            i += 1
            j += 1
    else:
        if fpointer == -1 and max_idx != -1:
            fpointer = random.randint(0, max_idx)
        if isinstance(fpointer, int):
            fpointer = list(range(fpointer, min(fpointer + batch_size, len(data['arr_0']))))
        for col in useful_cols:
            while input_cols[i] != col:
                j += 1
                if 'str' in input_types[i]:
                    j += 1
                i += 1
            if 'str' in input_types[i] and 'id' not in input_cols[i]:
                cols[col + 'seq'] = data['arr_' + str(j)][fpointer].astype(np.int32)
                cols[col + 'mask'] = data['arr_' + str(j + 1)][fpointer].astype(np.int32)
                j += 1
                if use_segment_id:
                    cols[col + 'seg'] = data['arr_' + str(j + 1)][fpointer].astype(np.int32)
                    j += 1
            elif input_types[i] == 'int':
                cols[col] = data['arr_' + str(j)].astype(np.int32)
            elif input_types[i] == 'str':
                cols[col] = data['arr_' + str(j)]
            else:
                cols[col] = data['arr_' + str(j)].astype(np.float)
            i += 1
            j += 1
    return cols


def text_to_tokenids(filename, fpointer, batch_size, berttokenizer, 
                    input_cols, input_types, useful_cols, max_idx = -1, 
                    max_n_seq = 12, max_n_seqdoc = 12, use_segment_id=False, 
                    out_npz=False, out_npz_file=None, pbar=None, lock=None):
    results = dict()
    with open(filename, 'r', encoding = 'utf-8') as fp:
        if fpointer == -1 and max_idx != -1:
            fpointer = random.randint(0, max_idx)
            fp.seek(fpointer)
            fp.readline()
        else:
            fp.seek(fpointer)
        cnt = 0
        while True:
            s = fp.readline().strip('\n\r')
            if s == '':
                break
            tokens = s.split('\t')
            for col in useful_cols:
                pos = input_cols.index(col)
                token = tokens[pos]
                if col not in results.keys():
                    if 'id' in col or 'str' not in input_types[pos]:
                        results[col] = []
                if 'str' in input_types[pos] and 'id' not in col:
                    if col + 'seq' not in results.keys():
                        results[col + 'seq'] = []
                        results[col + 'mask'] = []
                    if use_segment_id and col + 'seg' not in results.keys():
                        results[col + 'seg'] = []
                    if 'q' in col:
                        seq_len = max_n_seq
                    elif 'k' in col or 'd' in col:
                        seq_len = max_n_seqdoc
                    q_tokens = berttokenizer.tokenize(token)[0:seq_len]
                    qseq = berttokenizer.convert_tokens_to_ids(q_tokens)
                    qmask = [1] * len(qseq)
                    if use_segment_id:
                        qseg = [1] * len(qseq)
                        add = 0
                        for i in range(len(qseq)):
                            t = qseq[i]
                            qseg[i] += add
                            if t == berttokenizer.sep_token_id:
                                add += 1
                    qseq = qseq + [0] * (seq_len - len(qseq))
                    qmask = qmask + [0] * (seq_len - len(qmask))
                    results[col + 'seq'].append(qseq)
                    results[col + 'mask'].append(qmask)
                    if use_segment_id:
                        qseg = qseg + [0] * (seq_len - len(qseg))
                        results[col + 'seg'].append(qseg)
                elif input_types[pos] == 'int':
                    results[col].append(int(token))
                elif input_types[pos] == 'str':
                    results[col].append(str(token))
                else:
                    results[col].append(float(token))
            cnt += 1
            if cnt == batch_size:
                break

    for k in results.keys():
        results[k] = np.asarray(results[k])
    
    offset = 0
    if out_npz:
        out_results = dict()
        for key in results:
            print(key)
            if key.endswith('seq'):
                out_results['arr_'+str(input_cols.index(key[0:-3])+offset)] = results[key]
            elif key.endswith('mask'):
                offset += 1
                out_results['arr_'+str(input_cols.index(key[0:-4])+offset)] = results[key]
            elif key.endswith('seg'):
                offset += 1
                out_results['arr_'+str(input_cols.index(key[0:-3])+offset)] = results[key]
            else:
                out_results['arr_'+str(input_cols.index(key)+offset)] = results[key]
        del results
        np.savez(out_npz_file, **out_results)
        #with lock:
        #pbar.update(f"pid:{mp.current_process().pid} result:{out_npz_file}")
        print(f"pid:{mp.current_process().pid} result:{out_npz_file}")
        return mp.current_process().pid, out_npz_file
    return results


def file_reader(type):
        if 'npz' in type:
            def npz_reader(filename, get_data=False):
                ori_data = np.load(filename)
                data = dict()
                l = 0
                for key in ori_data.keys():
                    data[key] = ori_data[key]
                    if l == 0:
                        l = len(data[key])
                    if not get_data:
                        return l
                return data, l
            return npz_reader
        else:
            raise RuntimeError("only support npz!!")
  

def split_text_file(inputfile, partitions, outputdir):
    """
    splits a large text file (inputfile) to smaller parts (defined by partitions) and places
    them in outputdir
    """
    from filesplit.split import Split
    split = Split(inputfile, outputdir)
    partition_size = os.path.getsize(inputfile) / partitions
    split.bysize(partition_size, newline=True)


def text_to_npz(inputfile, maxcntperpartition, outdir,
                num_workers = None):
    """Process a file in frames using multiple workers
    Args:
    filename: the file to process
    processes: the size of the pool
    outpudir, input_cols, input_type, useful_cols,
    """
    berttokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    input_cols = ['qid','query','did','doc']
    input_types = ['int','str','int','str']
    useful_cols = ['qid','query','did','doc']
    max_n_seq = 5
    max_n_seqdoc = 25
    if num_workers is None:
        num_workers = mp.cpu_count()
    chunks = parse_textfile(maxcntperpartition, inputfile, drop_last=False)
    #with mp.Manager() as manager:
    #    lock = manager.Lock()
    chunks = [(inputfile, chunks[idx], maxcntperpartition, berttokenizer, 
                input_cols, input_types, useful_cols, -1, 
                max_n_seq, max_n_seqdoc, False, True, outdir+'/'+str(idx)+'.npz') 
                for idx in range(0,len(chunks))]
    pool = mp.Pool(num_workers)
    pool.starmap(text_to_tokenids, chunks, chunksize=1)
    # pbar = tqdm(pool.starmap(text_to_tokenids, chunks, chunksize=1), total=len(chunks))
    # for result in pbar: # return a list of collected outputs (returns from the function)
    #     pbar.update(f"pid:{result[0]} result:{result}")


if __name__=='__main__':
    os.makedirs('../../../data/orcas_npz_splits/', exist_ok=True)
    text_to_npz('C:\\Users\\thumm\\Documents\\machineLearning\\nlp\\data\\orcas_pre-processed.tsv', 
                752960, 
                '../../../data/orcas_npz_splits',
                2)
