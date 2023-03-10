import tensorflow as tf

#多元框损失的计算及其辅助计算函数
class MultiboxLoss(object):
    """
    # 参数
        num_classes: 种类数
        alpha: L1-smooth损失的权重
        neg_pos_ratio: 负样本到正样本的最大损失率
        background_label_id:背景标签的id
        negatives_for_hard: 负样本数量


    """
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('没有可用的背景标签')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
        
#计算L1-smooth损失的权重
    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

#计算softmax损失
    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),axis=-1)
        return softmax_loss

#计算多元框损失
    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        #所有先验框的损失
        conf_loss = self._softmax_loss(y_true[:, :, 4:-8],y_pred[:, :, 4:-8])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],y_pred[:, :, :4])

        # 正样本损失
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],axis=1)

        # 负样本损失（只有置信度）
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg,
                                [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) + tf.reshape(indices, [-1]))
        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,[batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # 正负样本损失总和
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss
