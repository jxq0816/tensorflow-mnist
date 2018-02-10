#coding=utf-8
import tensorflow as tf
import input_data
#---------------------------定义变量-------------------------------------
# 通过操作符号变量来描述这些可交互的操作单元
# x一个占位符placeholder,我们在TensorFlow运行计算时输入这个值
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量，我们用2维的浮点数张量来表示这些图
# 这个张量的形状是[None，784]（这里的None表示此张量的第一个维度可以是任何长度的）
print("define model variable ");
x = tf.placeholder("float", [None, 784])

# 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中
# 它们可以用于计算输入值，也可以在计算中被修改
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

# W:权重
# 注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
W = tf.Variable(tf.zeros([784,10]))

# b:偏移量
# b的形状是[10]，所以我们可以直接把它加到输出上面
b = tf.Variable(tf.zeros([10]))

#---------------------------定义模型-------------------------------------
print("define model ");
# 用tf.matmul(​​X，W)表示x乘以W
# 这里x是一个2维张量拥有多个输入
# 然后再加上b，把和输入到tf.nn.softmax函数里面
# 一行代码来定义softmax回归模型，y 是我们预测的概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)

#---------------------------训练模型-------------------------------------
print("define train model variable ");
# y' 是实际的概率分布，添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None,10])

#计算交叉熵，交叉熵是用来衡量我们的预测用于描述真相的低效性
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化我们创建的变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("start to train model")
for i in range(1000):
  #print(i)
  # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ---------------------------评估模型-------------------------------------
print("review model")
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))