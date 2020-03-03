from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import pickle
import keras
from Parking import Parking
cwd = os.getcwd()

def keras_model(weights_path):
    model = load_model(weights_path)
    return model
def img_test(test_images,final_spot_dict,model,class_dictionary):
    for i in range (len(test_images)):
        predicted_images = park.predict_on_image(test_images[i],final_spot_dict,model,class_dictionary)
def video_test(video_name,final_spot_dict,model,class_dictionary):
    name = video_name
    cap = cv2.VideoCapture(name)
    park.predict_on_video(name,final_spot_dict,model,class_dictionary,ret=True)
    
    
if __name__ == '__main__':
    weights_path = 'car1.h5'
    video_name = 'parking_video.mp4'
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'
    park = Parking()
    img = cv2.imread('test_images/scene1410.jpg')
    rect_images, rect_coords=park.image_process(img)
    rect_coords=park.rect_coords_modify(rect_coords)
    new_image, spot_dict = park.draw_parking(img, rect_coords)
    park.save_images_for_cnn(img, spot_dict)
    model = keras_model(weights_path)
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    img_test(test_images, spot_dict, model, class_dictionary)



    