from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np



def ssdLoss(hard_neg_ratio=3,alpha=1):

    def l1_smooth_loss(y_true,y_pred):
        x = K.abs(y_true - y_pred)
        x = tf.where(tf.less(x, 1.0), 0.5 * x ** 2, x - 0.5)
        return K.sum(x, axis=-1)


    def loss(y_true, y_pred):

        positives = K.cast(K.max(y_true[:,:,1:-12], axis=-1), dtype='float32') #batch size n boxes
        negatives = K.cast(y_true[:,:,0], dtype='float32')
        batch_size = K.shape(y_pred)[0] # batch size
        n_boxes = K.shape(y_pred)[1]

        ######## localization loss
        lloc = l1_smooth_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])
        lloc = K.sum(lloc*positives,axis=-1)  ## only penalizes predictions from positive matches

        ####### confidence loss

        classification_loss = categorical_crossentropy(y_true[:,:,:-12],y_pred[:,:,:-12])
        lconf = classification_loss*negatives

        n_positives=K.sum(positives,axis=-1) #batch size 
        n_negatives=K.sum(negatives,axis=-1) #batch size



        # def conf_zero():
        #     return(K.zeros([batch_size]))

        def conf_loss():

            lconf_batch = []

            for i in range(batch_size):
                if(n_negatives[i] < hard_neg_ratio*n_positives[i]):
                    lconf_batch.append(K.sum(lconf[i]))
                else:
                    values,indices = tf.nn.top_k(lconf[i],hard_neg_ratio*n_positives[i])
                    lconf_batch.append(K.sum(values))

            lconf_batch = K.constant(lconf_batch)
            return lconf_batch


        # lconf_batch = tf.cond(K.equal(K.sum(lconf), K.constant(0)), conf_zero, conf_loss)

        n_negative_keep = tf.minimum(hard_neg_ratio* K.sum(n_positives), K.sum(n_negatives))



        def f1():
            return tf.zeros([batch_size])
        def f2():
            neg_class_loss_all_1D = tf.reshape(lconf, [-1])
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=K.cast(n_negative_keep,dtype='int32'),
                                          sorted=False) 
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))
            negatives_keep = K.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]),dtype="float32") 
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) 
            return neg_class_loss


        lconf_batch = tf.cond(K.equal(K.sum(lconf), K.constant(0)), f1, f2)


        #### final loss

        n_positives = K.maximum(n_positives, 1)
        total_loss = (lconf_batch + alpha*lloc)/n_positives

        return total_loss

    return loss
