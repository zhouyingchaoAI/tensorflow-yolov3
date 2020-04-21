import requests
import numpy as np
import struct


URL1 = "http://192.168.60.88:8000/api/infer/coco_ensemble/1"
img_size = (416, 416)
img = np.ones((3, ) + img_size)


input_name = 'input_1'
output_name = 'output_1'
dims = [1]
dim_format = ''.join(['dims: %d' % dim for dim in dims])

img = img.ravel()
data = img.tolist()
length = len(data)
buf = struct.pack('%sf' % len(data), *data)



nv_inferreq = (
        ['batch_size: 1 '] +
        [' input {{ name: "{}"}}'.format(input_name)] +
        [' output {{ name: "{}" }}'.format(output_name)])
nv_inferreq = ''.join(nv_inferreq)
headers = {
    'Expect': '',
    'Content-Type': 'application/octet-stream',
    'NV-InferRequest': nv_inferreq}



r = requests.post(URL1, headers=headers, data=buf)

print(r.text)