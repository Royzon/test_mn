import mxnet as mx
import numpy as np
import logging

logging.getLogger().setLevel(logging.DEBUG)

# 设置随机数种子
mx.random.seed(42)
# 批大小
batch_size = 10

# 数据准备

# 训练集数据
train_data  = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])

# 验证集数据
eval_data  = np.random.uniform(0, 1, [20, 2])
eval_label = np.array([eval_data[i][0] + 2 * eval_data[i][1] for i in range(20)])

# 测试集数据
test_data  = np.random.uniform(0, 1, [10, 2])
test_label = np.array([test_data[i][0] + 2 * test_data[i][1] for i in range(10)])

# 准备好的数据需要放入迭代器中
# 这里使用的是NDArrayIter迭代器，事实上，MXNet还提供了其他迭代器可以使用
train_iter = mx.io.NDArrayIter(
    train_data,
    train_label,
    batch_size,
    shuffle=True ,
    label_name='lin_reg_label'
    )

eval_iter = mx.io.NDArrayIter(
    eval_data,
    eval_label,
    batch_size,
    shuffle=False,
    label_name='lin_reg_label'
    )

test_iter = mx.io.NDArrayIter(
    test_data,
    test_label,
    batch_size,
    shuffle=False,
    label_name='lin_reg_label'
    )

# 定义神经网络模型

# 一个模型需要输入层、隐藏层、输出层
# MXNet定义模型使用使用sym/symbol来定义
# 输入层通常是var/Variable
# 输出层的后缀是Output，输出层包括了损失层

# 定义输入层
X = mx.sym.var('data')          # X = mx.symbol.Variable('data') 两者等价
Y = mx.sym.var('lin_reg_label') # Y = mx.symbol.Variable('lin_reg_label') 两者等价
# 定义隐藏层
net = mx.sym.FullyConnected(name='fc1', data=X, num_hidden=1)
# 定义输出层
net = mx.sym.LinearRegressionOutput(name='lro', data=net, label=Y)

# 将一层一层的神经元拼凑成模型
model = mx.mod.Module(
    symbol=net,
    data_names=['data'],
    label_names=['lin_reg_label']
    )

# 模型可视化
# shape = {'data':(batch_size, 1, 1, 2)}
# mx.viz.plot_network(symbol=net, shape=shape).view()   # 显示模型结构图
# mx.viz.print_summary(symbol=net, shape=shape)         # 显示模型参数

#训练神经网络

model.fit(
    train_iter,                 # 设置训练迭代器
    eval_data=eval_iter,        # 设置验证迭代器
    num_epoch=20,               # 训练轮数
    eval_metric='mse',          # 损失函数
    optimizer='sgd',            # “随机梯度下降”求解器
    optimizer_params={
        'learning_rate': 0.01,  # 学习率
        "momentum": 0.9         # 惯性动量
        }
    )

# 在测试集上测试网络

metric = mx.metric.MSE()                        # 设置评价函数
mse = model.score(test_iter, metric)            # 测试并评价
print( "\ntest's mse: " + str(mse[0][1]) )      # 打印测试结果
