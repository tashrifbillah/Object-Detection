# Object-Detection
Object Detection in an Image Applying Deep Q-Network

Object detection refers to drawing a bounding box around the most specific location of an object in an image. There is a
reinforcement learning agent that autonomously performs this task after adequate training. 
The algorithm can be broadly classified into three steps- CNN, DQN, and SVM. In fact the work is a reproduction of
[this](http://slazebni.cs.illinois.edu/publications/iccv15_active.pdf) paper. However, all the codes are written by the
Tashrif Billah and Alexander Loh. Please see the project report for detailed description.

# CNN: Convolutional Neural Network
A pretrained CNN extracts feature from the proposed region.

# DQN: Deep-Q Network
The DQN is built on top of the CNN that determines the optimal transformation of the bounding box
so that the desired object is detected in as less number of steps as possible.

# SVM: Support Vector Machine
A category specific SVM recongizes the detected object.
