import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy as sp
import xmltodict
from scipy import misc
import pickle


# Defining constants
ROOT_PATH = "VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
ANNOTATION_FOLDER = "Annotations"
IMAGE_FOLDER = "JPEGImages"
CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car",
           "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

CLASS_TO_LABEL = {class_name: label for (class_name, label) in zip(CLASSES, range(len(CLASSES)))}


# Defining functions for extracting relevant data from the VOC 2012 database
def extract_objects(xml):
    objects = xml['annotation']['object']
    return [objects] if isinstance(objects,xmltodict.OrderedDict) else objects

def extract_labels(xml):
    return [CLASS_TO_LABEL[object['name']] for object in extract_objects(xml)]


def extract_bounding_boxes(xml):
    return [tuple([int(round(float(object['bndbox'][key])))
                   for key in ['xmin', 'ymin', 'xmax', 'ymax']]) for object in extract_objects(xml)]



def format_data(data):

    data_tuples = [(value["image"],value["labels"],value["bounding_boxes"]) for key,value in data.items()]

    values = [(image, bounding_boxes,labels)
              for (image, subimages, labels, bounding_boxes) in data_tuples]

    return values


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

        labels = extract_labels(xml)

        data[name] = {
            "labels":labels
        }

        print("Processed %s" % name)


    formatted_data = format_data(data)

    rl_data = formatted_data


    for ((image,bounding_boxes,labels),i) in zip(rl_data,range(len(rl_data))):
        sp.misc.imsave("out_rl/"+str(i)+".png",image)
        print(str(i))


    bounding_boxes = [bounding_boxes for (image,bounding_boxes,labels) in rl_data]
    pickle.dump(bounding_boxes, open("bounding_boxes.p", "wb"),protocol=4)


    labels_rl = [labels for (image,bounding_boxes,labels) in rl_data]
    pickle.dump(labels_rl, open("labels_rl.p", "wb"),protocol=4)


process_data()
