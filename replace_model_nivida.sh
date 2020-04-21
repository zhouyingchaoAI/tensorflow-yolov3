sudo docker cp /home/zyc/tensorflow-yolov3-master/model/helmet/yolov3/1/saved_model.pb tfserver_gpu:/models/mobilenet_coco/1
sudo docker kill tfserver_gpu
sudo docker start tfserver_gpu
