import numpy as np
import os
import scipy as sp
from scipy import misc
import matplotlib.pyplot as plt
import time
from numpy import genfromtxt
import skimage
from skimage import io

from keras import backend as K
K.set_learning_phase(1)
from scipy import io

from keras.applications.vgg19 import VGG19

start_time = time.time()

boxes = genfromtxt('predicted_bounding_boxes.csv', delimiter=',')
indices= genfromtxt('predicted_image_indices.csv', delimiter=',')
label= genfromtxt('predicted_image_labels.csv', delimiter=',')

N = len(label)
# M= np.zeros((N,4096),dtype=float)

M= [ ]
non_zero_labels= [ ]
base_model = VGG19(weights='imagenet')
get_output = K.function([base_model.layers[0].input], [base_model.layers[23].output]) # 23 for VGG19, 20 for VGG16
base_model.save("vgg19.h5")

for i in range(N):

    I = skimage.io.imread(os.path.join("out_rl", "%i.png" % indices[i]))
    box = boxes[i,: ].astype(int)
    J = I[box[1]:box[3], box[0]:box[2]]

    (w,h,d) = J.shape
    if w == 0 or h == 0:
        L = np.zeros((224,224,3))
    else:
        L = sp.misc.imresize(J,(224,224,3))
        # print("Positive detection")

        S = L.reshape(1, 224, 224, 3)
        O = get_output([S])[0]
        M.append(O.flatten())
        non_zero_labels.append(label[i])

   
    Plug in the trained CNN Model to extract feature
    S = L.reshape(1,224,224,3)
    O = get_output([S])[0]
    
    M[i,: ]= O.flatten()

    print(i)

sp.io.savemat("man_feature.mat",{"cnn_features":M})
sp.io.savemat("man_names.mat",{"names":label})

print("--- Execution time: %s seconds ---" % (time.time() - start_time))

print("Wait")

