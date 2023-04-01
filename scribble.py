from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import logging

# import yaml
# import logging.config
# with open('configs/logging_config.yaml', 'r') as f:
#     config = yaml.safe_load(f.read())
#     logging.config.dictConfig(config)
#     logging.captureWarnings(True)

# def get_logger(name: str):
#     """Logs a message
#     Args:
#     name(str): name of logger
#     """
#     logger = logging.getLogger(name)
#     return logger


def getLogger(name='root', loglevel='INFO'):
  logger = logging.getLogger(name)

  # if logger 'name' already exists, return it to avoid logging duplicate
  # messages by attaching multiple handlers of the same type
  if logger.handlers:
    return logger
  # if logger 'name' does not already exist, create it and attach handlers
  else:
    # set logLevel to loglevel or to INFO if requested level is incorrect
    loglevel = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(loglevel)
    fmt = '%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s'
    fmt_date = '%Y-%m-%dT%T%Z'
    formatter = logging.Formatter(fmt, fmt_date)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



def init_logger():
    import os

    filename = 'C:\\Users\\thumm\\Documents\\machineLearning\\nlp\\test.log'
    filemode = 'w'
    if os.path.exists(filename):
        with open(filename, 'a') as fp:
            fp.writelines(['\n\n', 'this is new run', '\n\n'])
        filemode = 'a'
    
    node = 0
    local_rank = 1
    global_rank = 1

    log_prefix = "node:"+str(node)+" - local_rank:"+str(local_rank)+" - global_rank:"+str(global_rank)+" - "

    logging.basicConfig(level="DEBUG",
                        filename=filename,
                        filemode=filemode,
                        format='%(asctime)s - ' + log_prefix + '%(process)d - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    # logging.debug('This is a debug message')
    # logging.info('This is an info message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')

class dummydataset(Dataset):
    def __init__(self):
        self.epoch = 0
        self.step = 0
    
    def __iter__(self):
        for i in range(0, len(self)):
            self.step += 1
            yield i
        self.epoch += 1
    
    def __getitem__(self, idx):
        return idx
    
    def __len__(self):
        return 50

class DataIter(Dataset):
    data_np = None
    data = None
    
    def __init__(self):
        self.data_np = np.array([x for x in range(2400000)])
        self.data = [x for x in range(2400000)]  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.array([data], dtype=np.int64)
        return data


class dummy(Dataset):
    fileinfo = []
    batch_offsets = []
    def __init__(self, fileinfo, batch_offsets):
        self.fileinfo = fileinfo
        self.batch_offsets = batch_offsets
    
    def __len__(self):
        return len(self.batch_offsets)

    def __getitem__(self, i):
        fid = self.batch_offsets[i][0]
        oid = self.batch_offsets[i][1]
        res = (self.fileinfo[0][fid], oid)
        print("getitem call--------------------------------------------------------")
        print("fileinfo ref count", sys.getrefcount(self.fileinfo))
        print("batch_offsets ref count", sys.getrefcount(self.batch_offsets))
        for x in self.batch_offsets:
            print("batch_offsets: ", x, "[0] ref count: ", sys.getrefcount(x[0]))
            print("batch_offsets: ", x, "[1] ref count: ", sys.getrefcount(x[1]))
        for x in self.fileinfo[0]:
            print("fileinfo: ", x, " ref count: ", sys.getrefcount(x))
        return res


def tracefun():
    train_data = DataIter()
    train_loader = DataLoader(train_data, batch_size=30,
                          shuffle=False,
                          drop_last=True,
                          pin_memory=False,
                          num_workers=3)
    #print("memory consumed at start ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    #for epoch in range(0, 10):
    #pdb.set_trace()
    for x in iter(train_loader):
        print(x)
        break


def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    print('Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename))
    return


def loggernametest():
    logger.info("inside loggernametest")

def textchunk_to_tokenids(frame):
    print("processing frame "+str(frame[1])+" in process "+str(os.getpid()))
    time.sleep(2)

def frame_supplier(filename, maxcntperpartition):
        """A generator for frames"""
        f = open(filename)
        offset = 0
        frameid = 0
        while True:
            s = mmap.mmap(f.fileno(), 1024*1024*1024, access=mmap.ACCESS_READ, offset=offset)
            cnt = 0
            frame = []
            for line in iter(s.readline, b""):
                if cnt >= maxcntperpartition:
                    yield [frame, frameid]
                    del frame
                    frame = []
                    cnt = 0
                    frameid += 1
                    print("memory consumed after frameid ", frameid, " is ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                frame.append(line)
                cnt += 1
            if len(frame) != 0:
                print("memory consumed after frameid ", frameid, " is ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                yield [frame, frameid]
            s.close()
            del s
            gc.collect
            offset += 1024*1024*1024

if __name__ == '__main__':
    # logger.info(msg="this is inside helper.py")
    # loggernametest()

    import subprocess
    import os

    rootdir = 'C:\\Users\\thumm\\Documents\\machineLearning\\nlp\\data\\orcas-doctrain-top100_split10'
    for filename in os.listdir(rootdir):
        res = subprocess.run(['wc', '-l', os.path.join(rootdir, filename)], capture_output=True, text=True)
        print(res.stdout)


    # filename = "C:\\Users\\thumm\\Documents\\machineLearning\\nlp\\data\\orcas-doctrain-top100_split10\\orcas-doctrain-top100_1"
    # maxcntperpartition=327680
    # #text_to_npz(filename,maxcntperpartition)
    # frames = frame_supplier(filename, maxcntperpartition)
    # for x in frames:
    #     continue
    # print("hello world")
    # print("memory consumed after completing frame processing is ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    # tracer = trace.Trace(ignoredirs=[sys.prefix, sys.exec_prefix], trace=1, count=0)

    # tracer.run('tracefun()')

    # # make a report, placing output in the current directory
    # r = tracer.results()
    # r.write_results(show_missing=True, coverdir=".")

    # sys.settrace(trace_calls)
    # tracefun()

    # fileinfo = [["file1", "file2"], 3]
    # batch_offsets = [[0, 0], [0, 3], [0, 6], [1, 0], [1, 3], [1, 6]]

    # print("start-----------------------------------------------------------------------")
    # print("fileinfo ref count", sys.getrefcount(fileinfo))
    # print("batch_offsets ref count", sys.getrefcount(batch_offsets))
    # for x in batch_offsets:
    #     print("batch_offsets: ", x, "[0] ref count: ", sys.getrefcount(x[0]))
    #     print("batch_offsets: ", x, "[1] ref count: ", sys.getrefcount(x[1]))
    # for x in fileinfo[0]:
    #     print("fileinfo: ", x, " ref count: ", sys.getrefcount(x))

    # dummy1 = dummy(fileinfo, batch_offsets)

    # print("after dummy1 creation --------------------------------------------------------")
    # print("fileinfo ref count", sys.getrefcount(fileinfo))
    # print("batch_offsets ref count", sys.getrefcount(batch_offsets))
    # for x in batch_offsets:
    #     print("batch_offsets: ", x, "[0] ref count: ", sys.getrefcount(x[0]))
    #     print("batch_offsets: ", x, "[1] ref count: ", sys.getrefcount(x[1]))
    # for x in fileinfo[0]:
    #     print("fileinfo: ", x, " ref count: ", sys.getrefcount(x))

    # dummy1[3]

    # print("done-----------------------------------------------------------------------")
    # print("fileinfo ref count", sys.getrefcount(fileinfo))
    # print("batch_offsets ref count", sys.getrefcount(batch_offsets))
    # for x in batch_offsets:
    #     print("batch_offsets: ", x, "[0] ref count: ", sys.getrefcount(x[0]))
    #     print("batch_offsets: ", x, "[1] ref count: ", sys.getrefcount(x[1]))
    # for x in fileinfo[0]:
    #     print("fileinfo: ", x, " ref count: ", sys.getrefcount(x))