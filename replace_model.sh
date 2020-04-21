sudo docker cp /home/zyc/tensorflow-yolov3-master/model/helmet/yolov3/1/saved_model.pb tfserver:/models/mobilenet_coco/1
sudo docker kill tfserver
sudo docker start tfserver
