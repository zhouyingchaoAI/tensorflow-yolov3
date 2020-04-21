import tensorflow as tf
import numpy as np
import base64
import os
from core.yolov3 import YOLOV3
import core.utils as utils
from PIL import Image

input_size      = 416

output_dir = "models"
model_version = 1
pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=17.8879.ckpt-30"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    serialized_input_data = tf.placeholder(dtype=tf.string, name='input_data')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
    }
    input_data = tf.parse_example(serialized_input_data, feature_configs)
    jpegs = input_data['image/encoded']
    image_string = tf.reshape(jpegs, shape=[])

    image_tensor = tf.image.decode_image(image_string,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))

    image_input = image_tensor[np.newaxis, ...] / 255


model = YOLOV3(image_input, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

# Export inference model.
output_path = os.path.join(tf.compat.as_bytes(output_dir), tf.compat.as_bytes(str(model_version)))
print ('Exporting trained model to', output_path)

builder = tf.saved_model.builder.SavedModelBuilder(output_path)

classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(serialized_input_data)


scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(probabilities)

classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classify_inputs_tensor_info
          },
          outputs={
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  scores_output_tensor_info
          },
          method_name=tf.saved_model.signature_constants.
          CLASSIFY_METHOD_NAME))

predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': predict_inputs_tensor_info},
                outputs={'scores': scores_output_tensor_info
      },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
  ))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
                'predict_images': prediction_signature,
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,},
        legacy_init_op=legacy_init_op)

builder.save()
print ('Successfully exported model to %s' % output_dir)






