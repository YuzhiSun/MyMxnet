import pandas as pd
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
nn.Dense()
# 交换坐标轴 下面记录的是 如何将（batchsize，vocab_size，num_steps）变成 （num_steps,batchsize, vocab_size）
# numpy版本
a = np.arange(1,25).reshape(2,3,4)
print(a,'\n')
aT = np.transpose(a,[2,0,1])
# NDarray版本
b = nd.arange(1, 25).reshape(2,3,4)
bT = nd.transpose(b, [2, 0, 1])
print(aT)
for metricb in bT:
    print(metricb)
for metric in aT:
    print(metric)