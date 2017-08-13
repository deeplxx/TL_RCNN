import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np


# prepro img
#
def load_img(path):
    """ 加载图片 """
    img = tl.visualize.read_image(path)
    # img = Image.open(path)
    return img


def resize_img(img, size, save_path=None):
    """ 变换尺寸 """
    img = tl.prepro.imresize(img, size=size)

    if save_path:
        tl.visualize.save_image(img, save_path)

    return img


def model_alexnet(x, num_classes, is_train=True):
    with tf.variable_scope('ALEX'):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.Conv2d(net, 96, (11, 11), (4, 4), act=tf.nn.relu, name='conv1')
        # net = RCNN_SVM.layers.BatchNormLayer(net)
        net = tl.layers.MaxPool2d(net, (3, 3), (2, 2), padding='VALID', name='pool1')

        net = tl.layers.Conv2d(net, 256, (5, 5), act=tf.nn.relu, name='conv2')
        # net = RCNN_SVM.layers.BatchNormLayer(net)
        net = tl.layers.MaxPool2d(net, (3, 3), (2, 2), padding='VALID', name='pool2')

        net = tl.layers.Conv2d(net, 384, (3, 3), act=tf.nn.relu, name='conv3')
        net = tl.layers.Conv2d(net, 384, (3, 3), act=tf.nn.relu, name='conv4')
        net = tl.layers.Conv2d(net, 256, (3, 3), act=tf.nn.relu, name='conv5')
        # net = RCNN_SVM.layers.BatchNormLayer(net)
        net = tl.layers.MaxPool2d(net, (3, 3), (2, 2), padding='VALID', name='pool5')

        net = tl.layers.FlattenLayer(net, name='flatten')
        # net = RCNN_SVM.layers.BatchNormLayer(net, is_train=is_train, name='bn1')
        net = tl.layers.DropoutLayer(net, 0.5, is_fix=True, is_train=is_train, name='drop1')
        net = tl.layers.DenseLayer(net, 4096, act=tf.nn.tanh, name='tanh1')
        # net = RCNN_SVM.layers.BatchNormLayer(net, 0.8, is_train=is_train, name='bn2')
        net = tl.layers.DropoutLayer(net, 0.5, is_fix=True, is_train=is_train, name='drop2')
        net = tl.layers.DenseLayer(net, 4096, act=tf.nn.tanh, name='tanh2')
        # net = RCNN_SVM.layers.BatchNormLayer(net, is_train=is_train, name='bn3')
        net = tl.layers.DropoutLayer(net, 0.5, is_fix=True, is_train=is_train, name='drop3')
        net = tl.layers.DenseLayer(net, num_classes, act=tf.identity, name='softmax')

    return net


def split_data(x, y):
    perm = np.random.permutation(x.shape[0])
    mid = int(x.shape[0] * 0.9)
    idx_train = perm[: mid]
    idx_valid = perm[mid:]

    return x[idx_train], y[idx_train], x[idx_valid], y[idx_valid]


def train(x_data, y_data, num_classes, save_path):

    x_train, y_train, x_valid, y_valid = split_data(x_data, y_data)

    # train
    #
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y_ = tf.placeholder(tf.int64, (None, num_classes))
    net = model_alexnet(x, num_classes)
    y = net.outputs

    # define cost and metric
    cost = tl.cost.cross_entropy(y, tf.argmax(y_, 1), name='cost')
    correct_predic = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_predic, tf.float32))
    # y_op = tf.argmax(tf.nn.softmax(y))

    # define optimizer
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    # fine-tune
    #
    if os.path.isfile(save_path):
        print('Loading pre-train params...')
        tl.files.load_and_assign_npz(sess, save_path, net)
        # saver.restore(sess, 'net.ckpt')

    tl.utils.fit(sess, net, train_op, cost, x_train, y_train, x, y_,
                 acc, batch_size=32, n_epoch=300, print_freq=10,
                 X_val=x_valid, y_val=y_valid, eval_train=True)
    tl.files.save_npz(net.all_params, name=save_path, sess=sess)
    sess.close()


def predic_to_svm(load_path, imgs):
    """ 预测图片并将结果传递给svm分类器

    Args:
        load_path: 预训练参数
        imgs: fine-tune训练数据

    Returns:
        返回最后的4096维特征
    """
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, (None, 224, 224, 3))
    net = model_alexnet(x, 17, is_train=False)
    # y_op = tf.argmax(tf.nn.softmax(y), 1)

    tl.layers.initialize_global_variables(sess)  # 先init后load！！！！！！！！！！！！！！！！！

    # 加载并丢弃最后一层的两个参数[w, b]
    y = net.all_layers[-2]
    net.print_layers()

    params = tl.files.load_npz(name=load_path)
    params = params[:-2]
    tl.files.assign_params(sess, params, net)
    # net.print_params()

    features = sess.run(y, feed_dict={x: imgs})

    return features


def fine_tune(load_path, train_data, num_classes):
    """ fine_tune

    Args:
        load_path: -> list, 第一个存放fine_tune参数路径，第二个存放预训练参数路径
        train_data: -> list, 第一个存放样本，第二个存放标签
        num_classes: 样本类别数+1

    Returns:
        保存fine-tune参数
    """
    print('\nStart fine-tune...')
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, (None, 224, 224, 3))
    y_ = tf.placeholder(tf.float32, (None, num_classes))
    net = model_alexnet(x, num_classes)
    y = net.outputs

    # define cost and metric
    cost = tl.cost.cross_entropy(y, tf.argmax(y_, 1), name='cost')
    correct_predic = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_predic, tf.float32))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
    # y_op = tf.argmax(tf.nn.softmax(y))

    tl.layers.initialize_global_variables(sess)

    if os.path.isfile(load_path[0]):
        tl.files.load_and_assign_npz(sess, load_path[0], net)
    elif os.path.isfile(load_path[1]):
        params = tl.files.load_npz(name=load_path[1])
        params = params[:-2]
        tl.files.assign_params(sess, params, net)

    tl.utils.fit(sess, net, train_op, cost, train_data[0], train_data[1], x, y_, acc,
                 batch_size=32, n_epoch=100, print_freq=5)

    tl.files.save_npz(net.all_params, name=load_path[0], sess=sess)
    sess.close()



if __name__ == '__main__':
    img1 = load_img('image_0001.jpg')
    img1 = resize_img(img1, (224, 224))
    print(img1.shape)
    # img2 = load_img('image_0321.jpg')
    # img2 = resize_img(img2, (227, 227))
    # data_dir = 'd:/work/PycharmProject/source/jpg'

    # x, y = oxf17.load_data(one_hot=True)

    # train(x, y, 17)
    pred = predic_to_svm('model/net_params.npz', [img1])
    print(pred.shape)
    # print(pred[:10])
