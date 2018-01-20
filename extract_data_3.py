import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xmltodict
from scipy import io
from scipy import misc
import pickle


ROOT_PATH = "VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
ANNOTATION_FOLDER = "Annotations"
IMAGE_FOLDER = "JPEGImages"
CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car",
           "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
CLASS_TO_LABEL = {class_name: label for (class_name, label) in zip(CLASSES, range(len(CLASSES)))}

def extract_objects(xml):
    objects = xml['annotation']['object']
    return [objects] if isinstance(objects,xmltodict.OrderedDict) else objects

def extract_labels(xml):
    return [CLASS_TO_LABEL[object['name']] for object in extract_objects(xml)]


def extract_bounding_boxes(xml):
    return [tuple([int(round(float(object['bndbox'][key])))
                   for key in ['xmin', 'ymin', 'xmax', 'ymax']]) for object in extract_objects(xml)]


def process_data():

    annotation_directory_path = os.path.join(ROOT_PATH, ANNOTATION_FOLDER)
    image_directory_path = os.path.join(ROOT_PATH, IMAGE_FOLDER)

    names_and_image_paths_and_xml_paths = [
        (os.path.splitext(filename)[0],join(image_directory_path, os.path.splitext(filename)[0] + ".jpg"), join(annotation_directory_path, filename))
        for filename in listdir(annotation_directory_path)]

    names_and_image_paths_and_xml_paths = [(name,image_path, annotation_path)
                                           for (name,image_path, annotation_path) in names_and_image_paths_and_xml_paths
                                           if isfile(annotation_path)]


    data = {}
    for (name,image_path, annotation_path) in names_and_image_paths_and_xml_paths:

        xml = xmltodict.parse(open(annotation_path, 'rb'))
        image = misc.imread(image_path, mode='RGB')
        labels = extract_labels(xml)
        bounding_boxes = extract_bounding_boxes(xml)

        data[name] = {
            "image":image,
            "bounding_boxes":bounding_boxes,
            "labels": labels
        }

        print("Processed %s" % name)

    data_list = [(data[key]["image"],data[key]["bounding_boxes"][0],data[key]["labels"]) for (key,value) in data.items() if len(data[key]["labels"]) == 1]
    images, bounding_boxes, labels = tuple([[data_list[i][j] for i in range(len(data_list))] for j in range(3)])

    try:
        os.mkdir("out2")
    except:
        pass

    for image,i in zip(images, range(len(images))):
        sp.misc.imsave("out2/"+str(i)+".png",image)


    # np.savetxt("bb.csv",bounding_boxes,delimiter=',',newline='\n')
    # np.savetxt("labels.csv",labels,delimiter=',',newline='\n')
    pickle.dump(bounding_boxes, open("bounding_boxes.p", "wb"),protocol=4)
    pickle.dump(labels, open("labels_rl.p", "wb"),protocol=4)


process_data()
