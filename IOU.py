import numpy as np
# import skimage.io
import selectivesearch
import RCNN
import time


# IOU
def area_intersect(left_a, right_a, bottom_a, top_a,
                   left_b, right_b, bottom_b, top_b):
    assert left_a < right_a
    assert bottom_a < top_a
    assert left_b < right_b
    assert bottom_b < top_b

    x_intersect = left_a < left_b <= right_a or left_a <= right_b < right_a
    y_intersect = bottom_a < bottom_b <= top_a or bottom_a <= top_b < top_a
    if x_intersect and y_intersect:
        x_sorted = sorted([left_a, right_a, left_b, right_b])
        y_sorted = sorted([bottom_a, top_a, bottom_b, top_b])
        weight_intersect = x_sorted[2] - x_sorted[1]
        height_intersect = y_sorted[2] - y_sorted[1]

        area_intersect = weight_intersect * height_intersect
        return area_intersect
    else:
        return 0


def iou(vertice1, vertice2):
    area_i = area_intersect(vertice1[0], vertice1[2], vertice1[1], vertice1[3],
                            vertice2[0], vertice2[2], vertice2[1], vertice2[3])

    if area_i:
        area_1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
        area_2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])

        return area_i / (area_1 + area_2 - area_i)
    return 0


# clip
def clip_img(img, rect):
    right = rect[0] + rect[2]
    top = rect[1] + rect[3]
    return img[rect[0]: right, rect[3]: top + rect[3], :], [rect[0], rect[1], right, top]


# 从给定的datapath中返回images 和 对应的label
def load_train_proposals(datapath, num_class, threshold=0.5, svm=False,
                         save=False, save_path='dataset.npz'):
    """ 从给定的datapath中返回 images 和 label。其中 images 是 select=search 得到的候选框， label 是经过阈值比较后
    得到的标签，0表示背景。

    Args:
        datapath: txt文件，每一行包含 图片位置 类别 标签框（用来进行阈值比较）
        num_class: 类别数目
        threshold: IOU阈值
        svm: 是否喂给SVM
        save: 是否保存
        save_path: 保存路径

    Returns:
        训练数据list（images， labels）
    """
    train_list = open(datapath, 'r')
    images = list()  # 保存训练样本
    labels = list()  # 保存样本对应的标签
    value_error = 0
    i = 0
    t1 = time.time()

    for line in train_list:
        t2 = time.time()
        tmp = line.strip().split(" ")
        img = RCNN.load_img(tmp[0])
        if img is None:
            continue

        # img_lbl: [r,g,b,region]， regions{'rect'[left, bottom, w, h], 'size', 'labels'}
        img_lbl, regions = selectivesearch.selective_search(img, sigma=0.9, min_size=10)
        candidates = set()  # 保存select得到的候选框

        for r in regions:
            if r['rect'] in candidates:
                continue
            if r['rect'][2] <= 0 or r['rect'][3] <= 0:
                continue
            if r['size'] < 224:
                continue
            candidates.add(r['rect'])

            proposal_img, proposal_vertice = clip_img(img, r['rect'])
            try:
                img_array = RCNN.resize_img(proposal_img, (224, 224))
            except:
                value_error += 1
                continue
            images.append(img_array)

            ref_rect = tmp[2].split(',')  # 标签框？
            ref_rect_int = [int(x) for x in ref_rect]
            ref_rect_int = [ref_rect_int[0], ref_rect_int[1], ref_rect_int[0] + ref_rect_int[2],
                            ref_rect_int[1] + ref_rect_int[3]]
            iou_value = iou(ref_rect_int, proposal_vertice)

            idx = int(tmp[1])  # 属于的类别

            # 如果是SVM分类则label形式为int，若是fine-tune则为ont-hot
            if not svm:
                label = np.zeros(num_class + 1)
                if iou_value < threshold:  # 小于阈值则作为背景
                    label[0] = 1
                else:
                    label[idx] = 1
                labels.append(labels)
            else:
                if iou_value < threshold:
                    labels.append(0)
                else:
                    labels.append(idx)
        i += 1
        t3 = time.time() - t2
        print('{} image was proposaled!, cost {}s'.format(i, t3))
    t4 = time.time() - t1
    print('\ncount of value error is {}, total cost {}s'.format(value_error, t4))

    if save:
        try:
            np.savez(save_path, params=[images, labels])
        except Exception as e:
            print('np save error: {}'.format(e))
        else:
            print('[*]save SUCCESS!')
        # try:
        #     f = open(save_path, 'wb')
        # except Exception as e1:
        #     print('open file error: {}'.format(e1))
        # else:
        #     try:
        #         pickler = pickle.Pickler(f)
        #         pickler.fast = True
        #         pickler.dump((images, labels))
        #     except Exception as e:
        #         print('pickle dump error: {}'.format(e))
        #     finally:
        #         f.close()
        # finally:
        #     return images, labels

    # t1 = time.time()
    # images = np.array(images)
    # labels = np.array(labels)
    # t2 = time.time() - t1
    # print('\nconvert list to numpyarray cost {}s'.format(t2))
    #
    # return images, labels


def load_from_pkl(load_path):
    # f = open(load_path, 'rb')
    # x, y = pickle.load(f)
    # f.close()
    datasets = np.load(load_path)
    d = datasets['params']
    print('[*]load SUCCESS!')
    return d[0], d[1]


def img_proposal(img_path):
    """ 将给定的图片转换成 （框图片， 所在位置）的列表 """
    # img = skimage.io.imread(img_path)
    img = RCNN.load_img(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    images = list()
    vertices = list()

    for r in regions:
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        if r['size'] < 220:
            continue
        if r['rect'] in candidates:
            continue
        candidates.add(r['rect'])

        proposal_img, proposal_vertice = clip_img(img, r['rect'])
        resized_proposal_img = RCNN.resize_img(proposal_img, (224, 224))
        images.append(resized_proposal_img)
        vertices.append(proposal_vertice)

    return images, vertices


def nms(datasets, score, threshold):
    """ 非极大值抑制，输入预测同类别的框，输出保留下来的框的索引 """
    xmin = datasets[:, 0]
    ymin = datasets[:, 1]
    xmax = datasets[:, 2]
    ymax = datasets[:, 3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    order = score.argsort()[::-1]  # 每个框的得分按从大到小排序，order保存了序号

    keep = list()
    while order.size() > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        i_area = w * h
        # noinspection PyUnresolvedReferences
        iou_area = i_area / (areas[i] + areas[order[1:]] - i_area)

        idx = np.where(iou_area <= threshold)[0]
        order = order[idx + 1]  # iou_area的长度为 order[1:]即比order长度少了1，
        # 少了首个元素，iou_area中的idx对应都是order的idx-1

    return keep

# if __name__ == '__main__':
