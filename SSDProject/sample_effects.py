import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import UnivariateSpline


eye_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")


def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright

def contrast(img, alpha_val):
    img_bright = cv2.convertScaleAbs(img, alpha=alpha_val)
    return img_bright

def hue_shift(img, hue):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_shift = (hue - 180.0) / 2.0
    hue_shift = int(hue_shift)
    hsv_frame[:, :, 0] += hue_shift
    hsv_frame[:, :, 0] %= 180
    adjusted_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return adjusted_frame

def saturation_shift(img, saturation):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation_shift = int(saturation)
    hsv_frame[:, :, 1] += saturation_shift
    hsv_frame[:, :, 1] %= 255
    # s_shift = (saturation - 100.0) / 100.0
    # for j in range(0, hsv_frame.shape[0]):
    #     for i in range(0, hsv_frame.shape[1]):
    #         s = img[j, i, 1]
    #         s_plus_shift = s + 255.0 * s_shift
    #         if s_plus_shift < 0:
    #             s_plus_shift = 0
    #         elif s_plus_shift > 255:
    #             s_plus_shift = 255
    #         hsv_frame[j, i, 1] = s_plus_shift
    adjusted_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return adjusted_frame

def value_shift(img, value):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_shift = (value - 100.0) / 100.0
    # v_shift = int(value)

    # hsv_frame[:, :, 2] += v_shift
    # hsv_frame[:, :, 2] %= 255

    for j in range(0, hsv_frame.shape[0]):
        for i in range(0, hsv_frame.shape[1]):
            v = img[j, i, 2]
            v_plus_shift = v + 255.0 * v_shift
            if v_plus_shift < 0:
                v_plus_shift = 0
            elif v_plus_shift > 255:
                v_plus_shift = 255
            hsv_frame[j, i, 2] = v_plus_shift
    adjusted_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return adjusted_frame

def blur(img, value):
    blurred = cv2.blur(img, (value, value))
    return blurred

def opacity(img, val):
    alpha_channel = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    alpha_channel[:, :, 3] = val
    return alpha_channel

def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen

def pencil_sketch_grey(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_gray

def pencil_sketch_col(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_color

def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def resize_height(img, value):
    img = cv2.resize()

def crop(img, value, side):
    columns = img.shape[1]
    rows = img.shape[0]
    new_columns = int(value*columns)
    new_rows = int(value*rows)
    if side == 'left':
        cropped = img[:, columns-new_columns:]
    elif side == 'right':
        cropped = img[:, :new_columns]
    elif side == 'bottom':
        cropped = img[:new_rows, :]
    elif side == 'top':
        cropped = img[rows-new_rows:, :]
    return cropped

def image_resize(img, value, direction):
    columns =  img.shape[1]
    rows = img.shape[0]
    new_columns = int(value*columns)
    new_rows = int(value*rows)
    print(img.shape)
    if direction == 'width':
        resized = cv2.resize(img, (new_rows, new_columns))
    elif direction == 'height':
        resized = cv2.resize(img, (new_rows, columns))
    print(resized.shape)
    return resized

def emboss_image(image):
    emboss_kernel = np.array([[-1, 0, 0],[0,0,0],[0,0,1]])
    emboss_img = cv2.filter2D(src=image, ddepth=-1, kernel=emboss_kernel)
    return emboss_img

def median_blur(image):
    blur = cv2.medianBlur(src=image, ksize=9)
    return blur

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def invert(image):
    new_img = 255-image
    return new_img

def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def Summer(image):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

def Winter(image):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

def gradient(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5),np.uint8)
    grad = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
    return grad

def dialation(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)
    dia = cv2.dilate(img, kernel, iterations=1)
    return dia


if __name__ == "__main__":
    image = cv2.imread('D:\\Free-Guy.jpg')
    cv2.imshow('Original', image)
    cv2.waitKey(0)

    eye = eye_cascade.detectMultiScale(image)[0]
    eye_x, eye_y, eye_w, eye_h = eye
    glasses = plt.imread("sample1.png")
    glasses = cv2.resize(glasses,(eye_w+50,eye_h+55))
    image1 = image.copy()
    for i in range(glasses.shape[0]):
        for j in range(glasses.shape[1]):
            if glasses[i,j,3]>0:
                image1[eye_y+i-20,eye_x+j-23, :] = glasses[i,j,:-1]

    a1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a2 = bright(image, 60)
    a3 = sharpen(image)
    cv2.imshow('Grayscale', a1)
    cv2.imshow('Brightness change', a2)
    cv2.imshow('sharpened image', a3)
    cv2.imshow('cool image', image1)
    cv2.waitKey(0)


    cv2.destroyAllWindows()