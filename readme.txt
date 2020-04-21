1. UTIVA开发技巧
1.1 opencv编译依赖项（官网）
sudo aptitude install build-essential
sudo aptitude install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo aptitude install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
1.2 编译opencv静态库
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=install -DWITH_TBB=ON -DWITH_PNG=ON -DWITH_TIFF=ON -DWITH_JPEG=ON -DWITH_JASPER=1 -DBUILD_SHARED_LIBS=OFF -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSE=ON -DWITH_OPENMP=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF
make
1.3 boost库裁剪
boost源码目录下先生成./dist/bin/bcp
./dist/bin/bcp boost/date_time.hpp boost/bind.hpp boost/asio.hpp boost/thread.hpp boost/timer.hpp boost/foreach.hpp boost/thread/future.hpp boost/thread/mutex.hpp boost/thread/thread.hpp boost/asio/deadline_timer.hpp boost/archive/iterators/base64_from_binary.hpp boost/archive/iterators/binary_from_base64.hpp boost/archive/iterators/transform_width.hpp  boost/property_tree/ptree.hpp boost/property_tree/ini_parser.hpp boost/interprocess/shared_memory_object.hpp boost/interprocess/mapped_region.hpp boost/process.hpp boost/filesystem.hpp ./miniboost

2. 编译UTIVA
cd /path/to/UTIVA
./build-linux.sh
或者
./build-win64.bat

4. tensorflow模型训练环境（ubuntu已验证cpu版）
4.1 vscode安装
下载deb后安装：sudo apt-get install ./code_1.31.1-1549938243_amd64.deb (桌面快捷方式位于/usr/share/applications)
4.2 tensorflow + object detection模型训练环境安装(virtualenv方式)
git clone https://github.com/tensorflow/models.git 或者直接下载zip后进行解压
先验证后安装
python3 --version
pip3 --version
virtualenv --version
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv
创建虚拟环境
virtualenv --system-site-packages -p python3 ./venv
进入虚拟环境
source ./venv/bin/activate
虚拟环境中安装tensorflow
(venv)$pip install --upgrade pip
(venv)$pip install --upgrade tensorflow
虚拟环境中安装编译object detection
(venv)$sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
(venv)$pip install Cython
(venv)$pip install contextlib2
(venv)$pip install pillow
(venv)$pip install lxml
(venv)$pip install --default-timeout=100 jupyter
(venv)$pip install matplotlib
(venv)tensorflow/models/research/$wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
(venv)tensorflow/models/research/$unzip protobuf.zip
(venv)tensorflow/models/research/$./bin/protoc object_detection/protos/*.proto --python_out=.
(venv)tensorflow/models/research/$export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
退出虚拟环境
(venv)$deactivate
4.3 vscode + pyqt + tensorflow调试
$sudo apt-get install python3-pyqt5
vscode工作区设置中python.pythonPath = /path/to/venv/bin/python

5. tensorflow serving部署
5.1 docker安装(凝思/windows系统请参考docker官方文档)
$sudo apt-get update
$sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
$curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
$sudo apt-get install docker-ce docker-ce-cli containerd.io
5.2 tensorflow serving的docker镜像编译
$git clone https://github.com/tensorflow/serving
$cd serving
$sudo docker build --pull -t ut/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
$sudo docker build -t ut/tensorflow-serving --build-arg TF_SERVING_BUILD_IMAGE=ut/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile .
完成镜像编译后，主机的serving文件夹可删除
docker常用命令：
$sudo docker images #查看本地仓库镜像列表
$sudo docker run -d --restart=always --name tfserver -p 8500:8500 -p 8501:8501 ut/tensorflow-serving --model_config_file=/models/models.config 
$sudo docker cp -r /path/to/host/models/dir tfserver:/models/ #从主机向容器复制文件（容器必须已启动）
$sudo docker exec -it tfserver bash #进入容器查看、操作文件系统
$sudo docker commit tfserver ut/tensorflow-serving #将tfserver容器（文件系统已更改）固化为镜像ut/tensorflow-serving (镜像可覆盖)
$sudo docker stop tfserver #停止容器运行(需等待10秒)，kill命令可立即强行停止; start tfserver启动已创建的容器
$sudo docker rm tfserver #删除（由run创建的、已停止运行的）容器; rmi命令为删除镜像文件
$sudo docker logs tfserver #查看容器tfserver的运行日志
$sudo docker ps -a #查看所有已创建的容器; 不带-a则查看当前在运行状态的容器
$sudo docker save ut/tensorflow-serving -o /path/to/host/imagename.tar #导出镜像文件到主机
$sudo docker load -i /path/to/host/imagename.tar #从主机导入镜像文件

6. 模型训练超参数
6.1 样本不均衡：.config中classification loss使用focal loss（参考object detection文档TPU compatible detection pipelines）
6.2 使用GPU训练（1080TI/1060）时应增大batch_size以加快收敛速度，1080TI应能处理batch_size>=256，具体需试错
6.3 训练过程中打印消息：model_main.py中import部分之后即添加tf.logging.set_verbosity(tf.logging.INFO)默认100个迭代打印一次，将config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)改为config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, log_step_count_steps=k)可指定频率
6.4 样本/类别增减需重新训练，因每次训练时不会保留初始模型的目标检测能力
6.5 config文件的num_steps和model_main.py的num_train_steps是重复的，指定一个即可，如果都指定，config文件的会被覆盖
6.6 config文件的input_path可指定多个record文件：["filepath1", "filepath2",...]或指定整个目录，或指定目录下的通配符;coco数据集使用create_coco_tf_record.py转换为多个record文件
