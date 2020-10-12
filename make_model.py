from mxnet.gluon import nn
from mxnet import nd

class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(units=3, in_units=5)
print(dense.params)

class MyModel(nn.Block):
    def __init__(self, **kwargs):
        super(MyModel,self).__init__(**kwargs)
        self.line_weight = self.params.get('line_weight',shape=(2,1))

    def forward(self, X):
        Y = nd.dot(X,self.line_weight.data())

        return Y