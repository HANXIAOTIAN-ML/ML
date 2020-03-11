Parking_keras
====
parking_video.mp4是网上一个公开的停车场监控视频，我使用预处理算法得到车位位置并进行修正，获得空车位及占据车位训练集，然后使用Keras框架，采用迁移学习，调用预训练好的VGG16模型去掉顶层，加入全连接层，对训练集进行训练，对停车场摄像头传来的视频数据进行动态监测，使用模型获得每一帧的空车位及占据车位的位置，效果如下图所示，有6个空车位未检测出来，其他空车位、被占据车位均正确检测，对模型训练过程进行优化后，正确率可以进一步提升。
![Alt text](https://github.com/hxtuniverse/ML/blob/master/parking_keras/with_marking2.jpg)
