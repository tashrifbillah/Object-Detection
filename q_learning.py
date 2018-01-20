import random
import scipy as sp
from keras.layers import Dense, K
from keras.models import Sequential, load_model
import numpy as np
from scipy import misc
import pickle
import math
import matplotlib.pyplot as plt
from numba import jit
import queue
import sys
from scipy import io
import directories
import os
from collections import defaultdict
import tensorflow as tf
from collections import deque
import collections

target_update_interval = 1000
training = True
alpha = .1
experience_buffer_size = 2000
experience_sample_size = 15
gamma = .1
history_length = 10
epsilon_min = .1
epsilon_max = 1.0
epsilon_dec_steps = 5
epsilon_dec = (epsilon_max - epsilon_min) / epsilon_dec_steps
max_steps = 40


def crop_image(bb, image):

    w, h, d = image.shape
    bb = [int(math.floor(b)) for b in bb]
    bb[0] = max(bb[0], 0)
    bb[1] = max(bb[1], 0)
    bb[2] = min(bb[2], h)
    bb[3] = min(bb[3], w)
    cropped = image[bb[1]:bb[3], bb[0]:bb[2]]
    w, h, d = cropped.shape
    if w == 0 or h == 0:
        cropped = np.zeros((224, 224, 3))
    else:
        cropped = sp.misc.imresize(cropped, (224, 224, 3), interp='bilinear')
    return cropped


def get_unique_indices(labels):
    return [i for i in range(len(labels)) if len(labels[i]) == 1]


def flatten(arr):
    return [item for sublist in arr for item in sublist]


def intersection_area(boxA, boxB):
    dx = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
    dy = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def iou(boxA, boxB):
    inter = intersection_area(boxA, boxB)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


HUBER_DELTA = 1.0
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def initialize_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(4096 + 90,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(9, activation='linear'))
    model.compile(loss=smoothL1, optimizer='adam')
    return model


loss_arr = []


def fit(model, x, y):
    global loss_arr
    loss = model.train_on_batch(x, y)
    loss_arr.append(loss)
    if len(loss_arr) == 100:
        print("loss %s" % str(sum(loss_arr) / len(loss_arr)))
        loss_arr = []


class State:
    cnn_model = load_model(os.path.join(directories.data_dir, "vgg16.h5"))
    feature_extractor = K.function([cnn_model.layers[0].input], [cnn_model.layers[20].output])

    def __init__(self, history, bb, image):
        self.history = history
        self.bb = bb
        self.feature = State.compute_feature(history, bb, image)

    @staticmethod
    def compute_feature(history, bb, image):
        history_feature = State.get_history_feature(history)
        image_feature = State.get_image_feature(image, bb)
        feature = np.concatenate((image_feature, history_feature))
        return np.array([feature])

    @staticmethod
    def get_image_feature(image, bb):
        cropped = crop_image(bb, image)
        feature = State.feature_extractor([cropped.reshape(1, 224, 224, 3)])[0]
        return np.ndarray.flatten(feature)

    @staticmethod
    def get_history_feature(history):
        assert len(history) == history_length
        feature = np.zeros((90,))
        for i in range(history_length):
            action = history[i]
            if action != -1:
                feature[i * 9 + action] = 1
        return feature


def transform(bb, a):

    alpha = .2
    alpha_w = alpha * (bb[2] - bb[0])
    alpha_h = alpha * (bb[3] - bb[1])
    dx1 = 0
    dy1 = 0
    dx2 = 0
    dy2 = 0

    if a == 0:
        dx1 = alpha_w
        dx2 = alpha_w
    elif a == 1:
        dx1 = -alpha_w
        dx2 = -alpha_w
    elif a == 2:
        dy1 = alpha_h
        dy2 = alpha_h
    elif a == 3:
        dy1 = -alpha_h
        dy2 = -alpha_h
    elif a == 4:
        dx1 = -alpha_w
        dx2 = alpha_w
        dy1 = -alpha_h
        dy2 = alpha_h
    elif a == 5:
        dx1 = alpha_w
        dx2 = -alpha_w
        dy1 = alpha_h
        dy2 = -alpha_h
    elif a == 6:
        dy1 = alpha_h
        dy2 = -alpha_h
    elif a == 7:
        dx1 = alpha_w
        dx2 = -alpha_w

    bb = (bb[0] + dx1, bb[1] + dy1, bb[2] + dx2, bb[3] + dy2)
    bb = (
        min(bb[0], bb[2]),
        min(bb[1], bb[3]),
        max(bb[0], bb[2]),
        max(bb[1], bb[3]),
    )

    return bb


def trigger_reward(bb, true_bb):
    return 3 if iou(bb, true_bb) > .6 else -3


def transform_reward(bb, bbp, true_bb):
    return 1 if iou(bbp, true_bb) > iou(bb, true_bb) else -1


def get_q(s, model):
    return np.ndarray.flatten(model.predict(s.feature))


def select_action(s, true_bb, step, epsilon, action_values):

    if step == max_steps:
        a = 8

    else:
        if random.random() > epsilon:
            a = np.argmax(action_values)

        else:

            action_rewards = [transform_reward(s.bb, transform(s.bb, a_tmp), true_bb) for a_tmp in range(8)]
            action_rewards.append(trigger_reward(s.bb, true_bb))
            action_rewards = np.array(action_rewards)
            positive_action_indices = np.where(action_rewards >= 0)[0]

            if len(positive_action_indices) == 0:
                positive_action_indices = list(range(0, 9))
            a = np.random.choice(positive_action_indices)


    return a


def take_action(s, true_bb, a, image):

    if a == 8:
        sp = s
        r = trigger_reward(s.bb, true_bb)
        took_trigger = True

    else:

        bb = s.bb
        bbp = transform(bb, a)
        r = transform_reward(bb, bbp, true_bb)
        took_trigger = False
        historyp = s.history[1:]
        historyp.append(a)
        assert len(historyp) == history_length
        sp = State(historyp, bbp, image)

    return sp, r, took_trigger


def weights_from_errors(errors):

    sorted_inds = sorted(range(len(errors)),key=lambda x: errors[x])
    inv_ranks = [0]*len(errors)

    for i in range(len(inv_ranks)):
        inv_ranks[sorted_inds[i]] = 1.0/(len(inv_ranks)-i)


    return inv_ranks


def apply_experience(main_model, target_model,experience, experience_errors):

    weights = weights_from_errors(experience_errors)
    sample_inds = random.choices(range(len(experience)), k=experience_sample_size, weights = weights)
    sample = [experience[i] for i in sample_inds]

    targets = np.zeros((experience_sample_size, 9))

    for i in range(experience_sample_size):
        s, a, r, sp, done = sample[i]
        target = r

        if not done:
            target = compute_target(r, sp, target_model)
        targets[i, :] = get_q(s, main_model)
        targets[i][a] = target

    x = np.concatenate([s.feature for (s, a, r, sp, d) in sample])
    fit(main_model, x, targets)


def compute_target(r, sp, target_model):
    return r + gamma * np.amax(get_q(sp, target_model))


def copy_main_to_target_model_weights(main_model, target_model):
    weights = main_model.get_weights()
    target_model.set_weights(weights)

def q_learning_train(x, y, labels, epochs, main_model, target_model):

    epsilon = epsilon_max
    experience = collections.deque(maxlen=experience_buffer_size)
    experience_errors = collections.deque(maxlen=experience_buffer_size)
    total_steps = 0

    for epoch in range(epochs):

        print("epoch %i" % epoch)

        for xi, yi, l, data_index in zip(x, y, labels, range(len(x))):

            (width, height, d) = xi.shape
            initial_history = [-1] * history_length
            initial_bb = (0, 0, height, width)
            s = State(initial_history, initial_bb, xi)
            done = False
            total_reward = 0
            step = 0

            while not done:

                action_values = get_q(s, main_model)
                a = select_action(s, yi, step, epsilon, action_values)
                sp, r, done = take_action(s, yi, a, xi)
                step_experience = (s, a, r, sp, done)

                #add the experience and td-error to our buffer
                experience.append(step_experience)
                experience_errors.append(abs(action_values[a]-compute_target(r,sp,target_model)))

                #apply the experience
                apply_experience(main_model, target_model, experience, experience_errors)
                s = sp
                total_reward += r
                step += 1
                total_steps += 1

                #update the target Q-network
                if total_steps % target_update_interval == 0:
                    copy_main_to_target_model_weights(main_model,target_model)


            print("data_index %s" % data_index)
            print("reward %i" % total_reward)
            print("iou %f" % iou(s.bb, yi))

        if epoch < epsilon_dec_steps:
            epsilon -= epsilon_dec
            print("epsilon changed to %f" % epsilon)

    return main_model


def q_learning_predict(x,model):

    y = []
    for xi in x:

        (width,height,d) = xi.shape
        s = (0,0,height,width)
        history = [-1]*history_length
        history_feature = get_history_feature(history)
        done = False
        image_feature = get_image_feature(xi, s)
        feature = np.concatenate((image_feature, history_feature))

        for i in range(sys.maxsize):

            action_values = get_Q(model, feature)
            if i == max_steps-1:
                a = 8

            else:
                a = argmax(action_values)
            if a == 8:
                sp = s
                image_featurep = get_image_feature(xi, s)
                done = True

            else:
                sp = transform(s, a)
                image_featurep = get_image_feature(xi, s)

            history.append(a)
            history_featurep = get_history_feature(history)
            featurep = np.concatenate((image_featurep, history_featurep))
            s = sp
            feature = featurep

            if done:
                break
        y.append(s)

    return y


def load_data(filter_ratio, training_ratio, load_only_one=False):

    bbs = pickle.load(open(os.path.join(directories.data_dir, "bounding_boxes.p"), "rb"))
    print('loaded bbs')

    labels = pickle.load(open(os.path.join(directories.data_dir, "labels_rl.p"), "rb"))
    print('loaded labels')

    unique_indices = get_unique_indices(labels)
    indices_to_load = unique_indices[:int(len(unique_indices) * filter_ratio)]

    if load_only_one:
        indices_to_load = [unique_indices[0]]

    bbs = [bbs[i][0] for i in indices_to_load]
    labels = [labels[i] for i in indices_to_load]
    images = [sp.misc.imread(os.path.join(directories.data_dir, "out_rl", str(i) + ".png")) for i in indices_to_load]

    bbs_train = bbs[:int(len(bbs) * training_ratio)]
    bbs_test = bbs[int(len(bbs) * training_ratio):]

    labels_train = labels[:int(len(labels) * training_ratio)]
    labels_test = labels[int(len(labels) * training_ratio):]

    images_train = images[:int(len(images) * training_ratio)]
    images_test = images[int(len(images) * training_ratio):]


    if load_only_one:

        bbs_train = flatten([bbs for i in range(1000)])
        labels_train = flatten([labels for i in range(1000)])
        images_train = flatten([images for i in range(1000)])

    return bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load


def main():

    load_only_one = False
    filter_ratio = 1.0
    training_ratio = .8

    bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load = load_data(filter_ratio,
                                                                                                           training_ratio,
                                                                                                           load_only_one)

    print('images loaded')

    if training:

        main_model = initialize_model()
        weights = main_model.get_weights()
        target_model = initialize_model()
        target_model.set_weights(weights)
        model = q_learning_train(images_train, bbs_train, labels_train, 15, main_model, target_model)
        model.save("dqn.h5")

    else:

        model = load_model("dqn.h5")
        y = q_learning_predict(images_test, model)
        inds = range(int(len(images) * training_ratio), len(images))

        np.savetxt("predicted_bounding_boxes.csv", y, delimiter=',', newline='\n')
        np.savetxt("predicted_image_indices.csv", inds, delimiter=',', newline='\n')
        np.savetxt("predicted_image_labels.csv", labels_test, delimiter=',', newline='\n')

main()


