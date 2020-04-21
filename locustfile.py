from locust import HttpLocust, TaskSet, task, constant
# import cv2
# import base64
# import json
import random
# import numpy as np
URL1 = "http://192.168.60.73:8000/api/infer/coco/1"
URL2 = "http://192.168.60.73:8000/api/infer/pressplate/1"
URL3 = "http://192.168.60.73:8000/api/infer/device/1"
URL4 = "http://192.168.60.73:8000/api/infer/helmet/1"
# URL11 = "http://192.168.60.73:8901/v1/models/yolov3_helmet/versions/2:predict"
# URL21 = "http://192.168.60.73:8901/v1/models/yolov3_coco/versions/2:predict"
# URL41 = "http://192.168.60.73:8901/v1/models/yolov3_pressplate/versions/2:predict"

# URL1 = "http://192.168.60.73:8501/v1/models/mobilenet_coco:predict"
# URL2 = "http://192.168.60.88:8601/v1/models/mobilenet_coco:predict"
# URL3 = "http://192.168.60.88:8601/v1/models/mobilenet_utdevice:predict"
# URL4 = "http://192.168.60.88:8601/v1/models/mobilenet_pressplate:predict"
headers = {"content-type": "application/json"}

url_list = [URL1, URL2, URL3, URL4]
with open('/home/zyc/tensorflow-yolov3-master/VOC2019/JPEGImages/00001.jpg', 'rb') as f:
    data = f.read()

length = len(data)
batch_byte_size = length + 4
length_b = (length).to_bytes(4, byteorder='little')
data = length_b + data


class cocoTask(TaskSet):
    def on_start(self):
        self.url = url_list[random.randint(0, 3)]

    @task
    def test(self):

        print(self.url)
        input_name = 'inputs'
        output_names = ["detection_boxes", "detection_classes", "detection_scores", "num_detections"]
        dims = [1]
        dim_format = ''.join(['dims: %d' % dim for dim in dims])

        nv_inferreq = (
                ['batch_size: 1 '] +
                [' input {{ name: "{}" {} batch_byte_size: {} }}'.format(input_name, dim_format,
                                                                         batch_byte_size)] +
                [' output { name: "%s" }' % output_name for output_name in output_names])
        nv_inferreq = ''.join(nv_inferreq)
        headers = {
            'Expect': '',
            'Content-Type': 'application/octet-stream',
            'NV-InferRequest': nv_inferreq}
        response = self.client.post(self.url, data=data, headers=headers)


class WebsiteUser(HttpLocust):
    task_set = cocoTask
    wait_time = constant(1)
