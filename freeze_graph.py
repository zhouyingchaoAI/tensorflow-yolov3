#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
import numpy as np
import base64
from core.yolov3 import YOLOV3
import core.utils as utils
from PIL import Image

input_size      = 416

def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  image.set_shape((None, None, 3))
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  # image = tf.image.central_crop(image, central_fraction=0.875)
  # # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  # image = tf.image.resize_bilinear(
  #     image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
  # image = tf.squeeze(image, [0])
  # # Finally, rescale to [-1,1] instead of [0, 1)
  # image = tf.subtract(image, 0.5)
  # image = tf.multiply(image, 2.0)
  image_paded = tf.image.resize_image_with_pad(image, input_size, input_size)
  return image_paded/255

def _encoded_image_string_tensor_input_placeholder():
  """Returns input that accepts a batch of PNG or JPEG strings.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='input_data')

  def decode(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor

  return (batch_image_str_placeholder,
          tf.map_fn(
              decode,
              elems=batch_image_str_placeholder,
              dtype=tf.uint8,
              parallel_iterations=32,
              back_prop=False))


pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=18.1357.ckpt-30"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):

    input_data = tf.placeholder(dtype=tf.string, name='input_data')
    # image_tensor = tf.image.decode_image(input_data,
    #                                      channels=3)
    # image_tensor.set_shape((None, None, 3))
    #
    # frame_resize = tf.image.resize_images(image_tensor, [416, 416], method=0)
    #
    # image_input = frame_resize[np.newaxis, ...] / 255
    images = preprocess_image(input_data)
    # images = preprocess_image(serialized_tf_example)

    # images = tf.cast(images, tf.float32)


model = YOLOV3(images, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




