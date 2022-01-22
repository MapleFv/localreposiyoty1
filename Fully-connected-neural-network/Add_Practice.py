import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import math
from sklearn.model_selection import train_test_split
import os
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_eager_execution()  # 保证sess.run()能够正常运行

# # 数据集切分成训练集和测试集，占比为0.8：0.2
all_data = datasets.load_iris
# data1 = (all_data - all_data.mean()) / all_data.std()  # 数据标准化处理
x_data = datasets.load_iris().data  # 返回iris数据集所有输入特征
y_data = datasets.load_iris().target  # 返回iris数据集所有标签
train_X, test_X, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
train_X = np.array(train_X)
test_X = np.array(test_X)
# 这里可以将预测结果转换成了[0,1]或者[1,0]的对应形式，然后输出结果即为2维output_node=2也是可以的
en_one = OneHotEncoder()
train_Y = en_one.fit_transform(train_y.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
test_Y = en_one.fit_transform(test_y.reshape(-1, 1)).toarray()

input_node = 4  # 输入的feature的个数，也就是input的维度
output_node = 3  # 输出的分类标签0或者1
layer1_node = 10  # 隐藏层的节点个数，一般在255-1000之间，可以自行调整
batch_size = 5  # 批量训练的数据，batch_size越小训练时间越长，训练效果越准确（但会存在过拟合）实际用起来确实得改好多遍
learning_rate_base = 0.1  # 训练weights的速率η
regularzation_rate = 0.0001  # 正则力度
training_steps = 501  # 训练次数，这个指标需要类似grid_search进行搜索优化 #设定之后想要被训练的x及对应的正式结果y_
x = tf.placeholder(tf.float32, [None, input_node])
y_ = tf.placeholder(tf.float32, [None, output_node])

'''
tf.truncated_normal截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
stddev，标准差
'''
weight1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))  # input到layer1之间的线性矩阵
weight2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))  # layer1到output之间的线性矩阵
biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))  # input到layer1之间的线性矩阵的偏置
biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))  # layer1到output之间的线性矩阵的偏置


# 正向传播的流程，线性计算及激活函数relu的非线性计算得到result
def interence(input_tensor, weight1, weight2, biases1, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
    result = tf.matmul(layer1, weight2) + biases2
    return result


y = interence(x, weight1, weight2, biases1, biases2)
global_step = tf.Variable(0, trainable=False)  # global_step代表全局步数
# ----------------尝试输出global_step看一下结果-------------------- # 0
# init = tf.compat.v1.global_variables_initializer()
# with tf.compat.v1.Session() as sess:
#     sess.run(init)
#     print(sess.run(global_step))
# -----------------------------------------------------------------

'''
# 交叉熵，用来衡量两个分布之间的相似程度
为了防止过拟合，在交叉熵后面加上正则项
L1和L2正则化方法就是把权重的L1范数或L2范数加入到经验风险最小化的损失函数中（或者把二者同时加进去），
用来约束神经网络的权重，让部分权重为0（L1范数的效果）或让权重的值非常小（L2范数的效果），从而让模型变得简单，减少过拟合。得到的损失函数为结构风险最小化的损失函数。

# 1.x版本：
# regularzer = tf.contrib.layers.l2_regularizer(regularzation_rate)
# regularzation = regularzer(weight1) + regularzer(weight2)
'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
regularzation = regularzation_rate * tf.nn.l2_loss(weight1) + regularzation_rate * tf.nn.l2_loss(weight2)
loss = cross_entropy_mean + regularzation  # 损失函数为交叉熵+正则化

'''
我们用learning_rate_base作为速率η，来训练梯度下降的loss函数解
global_step代表全局步数
损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动加一的操作。学习率是伴随global_step的变化而变化的.
 最小化loss
'''
train_op = tf.train.GradientDescentOptimizer(learning_rate_base).minimize(loss, global_step=global_step)

'''
y是是真实值,y_result我们的输出结果，我们来找到y及y_result(比如[0.1，0.2])中最大值对应的index位置，判断y与y_result是否一致,返回bool类型
如果y与y_一致则为1，否则为0，mean正好为其准确率
cast()将correction转化为float类型
'''
correction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accurary = tf.reduce_mean(tf.cast(correction, tf.float32))
# 初始化环境，设置输入值，检验值
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
validate_feed = {x: train_X, y_: train_Y}
test_feed = {x: test_X, y_: test_Y}
print(train_X.shape, train_Y.shape)
'''
模型训练，每到1000次汇报一次训练效果
参数解释：batch_size = 200  # 批量训练的数据，batch_size越小训练时间越长，训练效果越准确（但会存在过拟合）实际用起来确实得改好多遍
参数解释：epoch = 500  # 训练次数，这个指标需要类似grid_search进行搜索优化
'''

# 模型训练，每到10次汇报一次训练效果
for i in range(training_steps):
    start = (i * batch_size) % len(train_X)
    end = start + batch_size
    xs = train_X[start:end]
    ys = train_Y[start:end]
    if i % 10 == 0:
        validate_accuary = sess.run(accurary, feed_dict=validate_feed)
        print('the times of training is %d, and the accurary is %s' % (i, validate_accuary))
    sess.run(train_op, feed_dict={x: xs, y_: ys})
