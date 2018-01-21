# Object-Detection
Object Detection in an Image Applying Deep Q-Network

Object detection refers to drawing a bounding box around the most specific location of an object in an image. There is a
reinforcement learning agent that autonomously performs this task after adequate training. 
The algorithm can be broadly classified into three steps- CNN, DQN, and SVM.

# CNN: Convolutional Neural Network
A pretrained CNN extracts feature from the proposed region.

# DQN: Deep-Q Network
The DQN is built on top of the CNN that determines the optimal transformation of the bounding box
so that the desired object is detected in as less number of steps as possible.

# SVM: Support Vector Machine
A category specific SVM recongizes the detected object.

In fact the work is performed as a research project. It replicates the results given in
[this paper](http://slazebni.cs.illinois.edu/publications/iccv15_active.pdf) and also implemented some modification.
However, all the codes are written by the Tashrif Billah and Alexander Loh. Please see the [project report](https://github.com/tashrifbillah/Object-Detection/blob/master/Tashrif_Billah_Object_Detection.pdf) for detailed description.
The algorithm was trained on Google Cloud GPU, courtesy of Electrical Engineering Department, Columbia University.

# Instruction for running code
Please see the following instruction for executing the project

1. Download the [VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

2. Run [extract_data.py](https://github.com/tashrifbillah/Object-Detection/blob/master/extract_data.py) that will extract
relevant information from the dataset i.e. image classes, indices, and bounding boxes.

3. Download the [pre trained CNN](https://drive.google.com/file/d/1_W-lSUUVBQsh0S2HM7bKeJcghtkVcWrL/view?usp=sharing) for feature extraction.

4. Follow the [tutorial](https://www.cs.columbia.edu/~smb/classes/f16/guide.pdf) to create your own virtual machine (VM) on Google Cloud GPU. Feel free to use any other GPU you might have access to.

5. Install necessary libraries on the VM.

6. Run [q_learning.py](https://github.com/tashrifbillah/Object-Detection/blob/master/q_learning.py) on the VM. This program will
train the DQN with weights for optimal bounding box deformation. This might take 6-10 hours depending on the speed of your system. However, a [pre trained DQN](https://drive.google.com/file/d/1ul3_Q9xG8K4MS79x2ECnTx11ocQG6BGy/view?usp=sharing) wait is already provided for testing.

7. Run [SVM_SCORE_VGG16.py](https://github.com/tashrifbillah/Object-Detection/blob/master/SVM_SCORE_VGG16.py) to extract feature from
the detected regios. A set of [predictions](https://drive.google.com/drive/folders/1ck8_2SMQl2b-eNtU4LYiWllSUIysb5bm?usp=sharing) are provided here for testing.

8. Now you are done with everything upto object recognition. Run [SVM_Model_Building](https://github.com/tashrifbillah/Object-Detection/blob/master/SVM_Model_Building.m) on MATLAB for 20 category specific SVM model training. A set of [example models](https://drive.google.com/drive/folders/1VqTUxozb1PijDERZ7iunFLw-IONvFXe8?usp=sharing) are provided here for testing.

9. At the last step, run [SVM_Scoring.m](https://github.com/tashrifbillah/Object-Detection/blob/master/SVM_Scoring.m) that will give you the overall performance of your algorithm. A set of [test features](https://drive.google.com/drive/folders/15T5kfvw44NaJHS7fZtV38bioo98Zhf7q?usp=sharing) are given for testing. 



