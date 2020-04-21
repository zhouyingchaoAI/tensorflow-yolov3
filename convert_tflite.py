import tensorflow as tf

graph_def_file = "yolov3_coco.pb"
input_array = ["input/input_data"]
output_array = ["pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

converter = tf.lite.TFLiteConverter.from_saved_model("model/yolov3/1")
converter.allow_custom_ops = True
converter.post_training_quantize = True
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("converted_model.tflite", "wb").write('model/yolov3/1')