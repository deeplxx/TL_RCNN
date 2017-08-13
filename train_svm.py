# coding=utf-8
import RCNN, IOU
from sklearn import svm
# import skimage.io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def generate_single_svm(one_class_trainfile):
    """ 生成训练svm的样本数据和标签 """
    trainfile_tmp = one_class_trainfile
    savepath = one_class_trainfile.replace('txt', 'pkl')  # 先判断有没有已经保存好的pkl

    if os.path.isfile(savepath):
        images, labels = IOU.load_from_pkl(savepath)
    else:
        images, labels = IOU.load_train_proposals(trainfile_tmp, num_class=2, threshold=0.3,
                                                  svm=True, save=True, save_path=savepath)

    return images, labels


def train_svm(train_file, params_path):
    train_list = os.listdir(train_file)
    svms = list()  # 保存训练好的svm分类器（每个类别训练一个）

    # 对每个文件列表（单一类）训练svm
    for train_txtfile in train_list:
        if 'pkl' in train_txtfile:
            continue
        x, y = generate_single_svm(train_file + train_txtfile)

        # 预测丢掉最后一层全连接，取倒数第二层的4096个输出作为特征输入给SVM
        svm_features = list()
        for i in x:
            feature = RCNN.predic_to_svm(params_path, i)
            svm_features.append(feature)

        clf = svm.LinearSVC()
        clf.fit(svm_features, y)
        svms.append(clf)

    return svms


if __name__ == '__main__':
    trainfile_folder = 'source/svm_train/'
    img_path = 'source/image_0080.jpg'
    load_path = ['model/fine_tune_params.npz', 'model/net_params.npz']


    if os.path.isfile('subrefine_dataset.pkl'):
        print('Loading Data...')
        x, y = IOU.load_from_pkl('subrefine_dataset.pkl')
    else:
        print('Reading Data...')
        IOU.load_train_proposals('source/subrefine_list.txt', 2, save=True, save_path='subrefine_dataset.pkl')
        x, y = IOU.load_from_pkl('subrefine_dataset.pkl')

    RCNN.fine_tune(load_path, (x, y), 3)

    print('Start train svm...')

    svms = train_svm(trainfile_folder, load_path[0])

    print('Train svm over...')

    imgs, vertices = IOU.img_proposal(img_path)  # 从一张图片得到若干个框图以及相应的坐标
    features = RCNN.predic_to_svm(load_path[0], imgs)  # 若干框图的特征

    results = list()
    results_labels = list()
    count = 0
    for f in features:  # 对每个框进行分类
        for i in svms:  # 对每个类别的分类器判断是否属于此类别（单个框可能被判断属于多个分类）
            pred = i.predict(f)
            if pred[0] != 0:
                results.append(vertices[count])
                results_labels.append(pred[0])
        count += 1

    # keep = IOU.nms(results, results_labels, 0.7)
    print('reuslt: \n', results)
    print('result_label: \n', results_labels)

    # img = skimage.io.imread(img_path)
    img = RCNN.load_img(img_path)
    fig, ax = plt.subplots(111, figsize=(6, 6))
    ax.imshow(img)
    for l, b, r, t in results:
        rect = mpatches.Rectangle((l, b), r-l, t-b, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

    ########################################################################################################
    # 分类完后还需要一个非极大值抑制来判断哪些框是没用重复的 ####################################################
    ########################################################################################################
