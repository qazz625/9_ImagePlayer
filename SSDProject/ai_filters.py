import cv2 as cv2
from PIL import Image, ImageEnhance
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h
    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
    return fc

def dog_filter(image):
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(gray, 1.09, 7)
    if len(fl) == 0:
        return []
    dog = cv2.imread('dog.png')
    dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in fl:
        frame = put_dog_filter(dog, image, x, y, w, h)
    return image

def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h
    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1
    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.40 * face_height)][x + j][k] = hat[i][j][k]
    return fc
    
def hat(image):
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(gray, 1.09, 7)
    if len(fl) == 0:
        return []
    hat = cv2.imread('hat.png')
    hat = cv2.cvtColor(hat, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in fl:
        frame = put_hat(hat, image, x, y, w, h)
    return image

def thug(image):
    eye_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")
    temp = eye_cascade.detectMultiScale(image)
    if len(temp) == 0:
        return []
    eye = temp[0]
    eye_x, eye_y, eye_w, eye_h = eye
    glasses = plt.imread("sample1.png")
    glasses = cv2.resize(glasses,(eye_w+50,eye_h+55))
    image1 = image.copy()
    for i in range(glasses.shape[0]):
        for j in range(glasses.shape[1]):
            if glasses[i,j,3]>0:
                image1[eye_y+i-20,eye_x+j-23, :] = glasses[i,j,:-1]
    return image1

def bgremove2(myimage):
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    finalimage = background+foreground
    return finalimage
