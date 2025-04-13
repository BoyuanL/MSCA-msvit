import torch
import os
import random
import numpy as np
import logging
import mxnet as mx

def np2tensor(arrays,device='gpu'):
    if isinstance(arrays, mx.nd.NDArray): #修改
        arrays = arrays.asnumpy()
    tensor=torch.from_numpy(arrays).type(torch.float)
    return tensor.cuda() if device=='gpu' else tensor


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def set_seed(seed):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)



def preprocess_input(x, data_format='channels_last', version=1):
    # preprocess_input函数，它接受一个图像张量（x）和（data_format：表示图像张量的维度顺序，缺省值为 'channels_last' 和 version：表示要使用的预处理方法的版本，缺省值为 1），并返回经过预处理后的图像张量。
    x_temp = np.copy(x) # 将输入图像张量 x 复制到一个新的 NumPy 数组 x_temp 中
    assert data_format in {'channels_last', 'channels_first'} # 如果指定的图像张量维度顺序 data_format 不在 'channels_last' 和 'channels_first' 中，则会抛出一个异常

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...] # 将 x_temp 沿着第二个维度逆序排列；这里用到了 Python 的切片语法，其中 : 表示选取整个维度，::-1 表示逆序选取，... 表示选取剩下的所有维度
            x_temp[:, 0, :, :] -= 93.5940 # x_temp 在第二个维度上选取第一个元素，然后对这个元素的所有元素都减去了一个常数 91.4953
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]  # 否则在最后一个维度（通道）上翻转图像
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp
