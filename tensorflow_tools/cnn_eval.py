#coding=utf-8

import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf
import time
from tensorflow_tools.data_explore import *

def get_data(path=None):
    # Load training and testing datasets.
    images, labels = load_data(path)

    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    # display_images_and_labels(images, labels)
    # display_label_images(images,labels, 26)
    # show_images_size(images)
    data = shringle_images(images, labels)
    return data


# 模型文件路径
model_path='cnn_model_train'
view_path_save = 'cnn_model_eval'
# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1


# 计算有多少类图片
#计算出来的
num_classes = 62


#
# print("logits: ", logits)
# print("mean_loss: ", mean_loss)

ROOT_PATH = 'F:/陶士来文件/tsl_python_project/model_datas'
# train_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Training/Training")
test_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Testing/Testing")

data=get_data(test_data_dir)
datas=data['images_a']
labels=data['labels_a']
#打乱顺序
num_example=datas.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
datas=datas[arr]
labels=labels[arr]

graph=tf.Graph()

with graph.as_default():

    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)

    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活#卷积核是对所有通道进行卷积操作的
    conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)

    predicted_labels = tf.arg_max(logits, 1)

    # 利用交叉熵定义损失
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )
    # 平均损失
    mean_loss = tf.reduce_mean(losses)

    # 定义优化器，指定要优化的损失函数
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses,global_step=step)

    correct_prediction = tf.equal(tf.cast(predicted_labels, tf.int32), labels_placeholder)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init = tf.initialize_all_variables()

    loss_summary = tf.summary.scalar('loss_eval', mean_loss)
    acc_summary = tf.summary.scalar('acc_eval', acc)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()


#
# print("logits: ", logits)
# print("mean_loss: ", mean_loss)


#打乱顺序
num_example=datas.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
datas=datas[arr]
labels=labels[arr]

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


with tf.Session(
        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True),
        graph=graph) as sess:

    #一开始是路径问题导致无法加载变量
    ckpt = tf.train.get_checkpoint_state(model_path)
    print('CKPT',ckpt)
    # if ckpt and ckpt.model_checkpoint_path:
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('开始加载变量')
        saver.restore(sess, ckpt.model_checkpoint_path)
    summary_writer = tf.summary.FileWriter(view_path_save, graph=sess.graph)

    n_epoch = 1
    batch_size = 128
    current_step = 0

    # sess.run(init)
    for epoch in range(n_epoch):
        print('开始预测')
        start_time = time.time()
        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(datas, labels, batch_size, shuffle=False):
            if current_step%1==0:
                err, ac ,summary= sess.run([mean_loss, acc,summary_op], feed_dict={datas_placeholder: x_val_a, labels_placeholder: y_val_a,dropout_placeholdr: 0})
                summary_writer.add_summary(summary, current_step)
                print("----- Step %d -- Loss %.5f -- acc %.5f" % (current_step, err, ac))
            current_step=current_step+1
'''
F:\陶士来文件\tsl_python_project\logo_recogniztion\tensorflow_tools>
tensorboard --logdir=train:cnn_model_train,eval:cnn_model_eval
'''
'''
   validation loss: 3.135902
   validation acc: 0.914683

'''




