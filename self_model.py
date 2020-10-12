from mxnet.gluon import nn, loss, data as gdata
from mxnet import nd, gluon,autograd
import mxnet as mx
class MyModel(nn.Block):
    def __init__(self, **kwargs):
        super(MyModel,self).__init__(**kwargs)
        self.line_weight = self.params.get('line_weight',shape=(2,1))

    def forward(self, X):
        Y = nd.dot(X,self.line_weight.data())

        return Y
X = nd.array([0,1])
model = MyModel()
batch_size = 8

X = nd.random.normal(scale=1,shape=(100,2))
X1 = nd.arange(0,200).reshape(100,2)
w1 = 1
w2 = 2
Y = w1 * X[:,0] + w2 * X[:,1]
dataset = gdata.ArrayDataset(X, Y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break
model.initialize()
trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': 10})

loss = loss.L2Loss()
for i in range(10):
    for X, Y in data_iter:
        with autograd.record():
            y_pre = model(X)
            l = loss(y_pre,Y)
        l.backward()
        trainer.step(batch_size)
    l = loss(model(X), Y)
    print('epoch %d, loss: %f' % (i, l.mean().asnumpy()))
    print(model.params['mymodel0_line_weight'].data())





