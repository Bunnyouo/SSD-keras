import numpy as np


class PriorBox():
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1, 0.1, 0.2, 0.2], clip=True, layer_shape=[8, 8], **kwargs):
        self.input_shape = layer_shape
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)

    def compute_default_box(self):
        layer_height = self.input_shape[0]
        layer_width = self.input_shape[1]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # 定义先验框的规格
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # 定义先验框的中心
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        # 生成数据集
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # 定义先验框的xmin, ymin, xmax, ymax
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        #定义变量
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        return prior_boxes


# 调用修改后的PriorBox类
img_size = (300, 300)
default_box_layer1 = PriorBox(img_size, 30, [], aspect_ratios=[2], layer_shape=(38, 38)).compute_default_box()
default_box_layer2 = PriorBox(img_size, 60, 114, aspect_ratios=[2, 3], layer_shape=(19, 19)).compute_default_box()
default_box_layer3 = PriorBox(img_size, 114, 168, aspect_ratios=[2, 3], layer_shape=(10, 10)).compute_default_box()
default_box_layer4 = PriorBox(img_size, 168, 222, aspect_ratios=[2, 3], layer_shape=(5, 5)).compute_default_box()
default_box_layer5 = PriorBox(img_size, 222, 276, aspect_ratios=[2, 3], layer_shape=(3, 3)).compute_default_box()
default_box_layer6 = PriorBox(img_size, 276, 330, aspect_ratios=[2, 3], layer_shape=(1, 1)).compute_default_box()
# 把各层的输出concatenate起来
default_box = np.concatenate((default_box_layer1, default_box_layer2, default_box_layer3, \
                              default_box_layer4, default_box_layer5, default_box_layer6), axis=0)
# 写入到pkl文件中
import pickle
pickle.dump(default_box, open("prior_boxes_ssd300", "wb"))
