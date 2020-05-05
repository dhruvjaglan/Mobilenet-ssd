from keras import backend as K
from keras.layers import Layer
import numpy as np

class AnchorBox(Layer):

  def __init__(self,img_size,scale,scale_next,aspect_ratios,n_boxes,variance=[1.0, 1.0, 1.0, 1.0], **kwargs): 
    self.img_height = img_size[1]
    self.img_width  = img_size[0]
    self.scale = scale
    self.scale_next = scale_next
    self.aspect_ratios =aspect_ratios
    self.n_boxes=n_boxes
    self.variance = variance ##### to do what is variance for 
    super(AnchorBox, self).__init__(**kwargs)

  def build(selimg_sizef, input_shape):
    # Create a trainable weight variable for this layer.
    super(AnchorBox, self).build(input_shape)  # Be sure to call this at the end

  def call(self, x):
    size= min(self.img_height,self.img_width)
    box_width=[]
    box_height=[]
    for ar in self.aspect_ratios:
      w = self.scale* np.sqrt(ar) * size
      h = self.scale*size / np.sqrt(ar)
      box_width.append(w)
      box_height.append(h)
      if ar == 1:
        w = size * np.sqrt(self.scale*self.scale_next*ar)
        h = size * np.sqrt(self.scale*self.scale_next/ar)
        box_width.append(w)
        box_height.append(h)
    if K.image_dim_ordering() == 'tf':
      batch_size, layer_height, layer_width, layer_channel= x._keras_shape
    else:
      batch_size, layer_channel, layer_height, layer_width= x._keras_shape


    step_x = self.img_width / layer_width
    step_y = self.img_height / layer_height

    cx = np.linspace(step_x/2, self.img_width-step_x/2, layer_width)
    cy = np.linspace(step_y/2, self.img_height-step_y/2,layer_height)

    centers_x, centers_y = np.meshgrid(cx,cy)
    centers_x = centers_x.reshape(-1, 1)
    centers_y = centers_y.reshape(-1, 1)
    boxes = np.concatenate((centers_x,centers_y),axis=1)
    boxes = np.tile(boxes,(1,2*n_boxes))
    boxes[:,::4] += box_width
    boxes[:,1::4] += box_height
    boxes[:,2::4] -= box_width
    boxes[:,3::4] -= box_height
    boxes[:, ::2] /= self.img_width
    boxes[:, 1::2] /= self.img_height
    boxes =boxes.reshape(-1,4)
    boxes = np.minimum(np.maximum(boxes, 0.0), 1.0)

    boxes1= np.copy(boxes).astype(np.float)

    boxes1[:,0]= (boxes[:,0]+boxes[:,2])/2  ### cen_x
    boxes1[:,1]= (boxes[:,1]+boxes[:,3])/2  #### cem y
    boxes1[:,2]= (boxes[:,0]-boxes[:,2])/2  ### w
    boxes1[:,3]= (boxes[:,1]-boxes[:,3])/2  #### h

    boxes1= np.reshape(boxes1, (layer_height,layer_width,n_boxes,4))
    variances_tensor = np.zeros_like(boxes1)

    variances_tensor += self.variances
    boxes1 = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

    ########### variance part I need to study

    boxes_tensor = np.expand_dims(boxes1, 0)
    boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))


    return boxes_tensor

  def compute_output_shape(self, input_shape):

    if K.image_dim_ordering() == 'tf':
      batch_size, layer_height, layer_width, layer_channel= input_shape
    else:
      batch_size, layer_channel, layer_height, layer_width= input_shape
    return (batch_size,layer_height,layer_width,self.n_boxes,8)