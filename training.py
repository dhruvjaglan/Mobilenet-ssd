import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
import numpy as np
from math import ceil
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger

from ssdloss import ssdLoss
from ssd import ssd300
from batchgenerator import BatchGenerator
from ssd_box_encode_decode_utils import SSDBoxEncoder

## constant

img_size=(300,300,3)
img_height=300
img_width=300
aspect_ratio_per_layer=[[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
  [1.0/3.0, 0.5, 1.0, 2.0, 3.0],[1.0/3.0, 0.5, 1.0, 2.0, 3.0],[0.5, 1.0, 2.0],[0.5, 1.0, 2.0]]
two_boxes_for_ar1 = True
coords = 'centroids'
n_classes = 20
variances = [1.0, 1.0, 1.0,1.0]  
limit_boxes = True 


model, predictor_sizes = ssd300(img_size=img_size,n_classes=n_classes)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = ssdLoss(hard_neg_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss)


train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])


VOC_2007_images_dir = '../VOC2007/JPEGImages_new/'
VOC_2007_annotations_dir = '../VOC2007/Annotations_new'

VOC_2007_train_image_set_filename = '../VOC2007/train.txt'
VOC_2007_val_image_set_filename      = '../VOC2007/val.txt'
VOC_2007_trainval_image_set_filename = '../VOC2007/trainval.txt'
VOC_2007_test_image_set_filename     = '../VOC2007/test.txt'

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                          image_set_filenames=[VOC_2007_train_image_set_filename],
                          annotations_dirs=[VOC_2007_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)


val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                  img_width=img_width,
                                  n_classes=n_classes,
                                  predictor_sizes=predictor_sizes,
                                  min_scale=0.2,
                                  max_scale=0.9,
                                  aspect_ratios_global=None,
                                  aspect_ratios_per_layer=aspect_ratio_per_layer,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  pos_iou_threshold=0.5,
                                  neg_iou_threshold=0.2,
                                  coords=coords)

batch_size = 32

train_generator = train_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=(0.5, 2, 0.5),
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(img_height, img_width, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

val_generator = val_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=(0.5, 2, 0.5),
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(img_height, img_width, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

n_train_samples = train_dataset.get_n_samples()
n_val_samples = val_dataset.get_n_samples()


def lr_schedule(epoch):
    if epoch <= 300:
        return 0.001
    else:
        return 0.0001

path = '/home/dhruv/Documents/Mobilenet-ssd/models'

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
   
checkpoint_path = path + "/ssd300_epoch-{epoch:02d}.h5"

checkpoint = ModelCheckpoint(checkpoint_path)
  
log_path = path + "/logs"

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)



terminate_on_nan = TerminateOnNaN()

callbacks = [checkpoint,csv_logger,learning_rate_scheduler,terminate_on_nan]

epochs = 10
intial_epoch = 0

history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=ceil(n_train_samples)/batch_size,
                                verbose=1,
                                initial_epoch=intial_epoch,
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_steps=ceil(n_val_samples)/batch_size,
                                callbacks=callbacks
                                )






