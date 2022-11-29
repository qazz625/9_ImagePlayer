from flask import Flask, render_template, Response, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from shear_rotate import *
from sample_effects import *
from seam_carve import *
from ai_filters import *
from PIL import Image
import numpy as np
import io
import cv2
import base64
import copy
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


prev_filter_request = ''
original_image = ''    
current_image = ''

def gen_frames():
    global img
    img = cv2.imread("sample1.png")
    buffer = cv2.imencode('.jpg', img)[1]
    frame = buffer.tobytes()
    # print(type(frame))
    # print(frame)
    return frame
    # cv2.imshow('sample image',img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

# gen_frames()

def get_base64(img):
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return img_base64



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/edit')
def filter_page():
    return render_template('play2.html')

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/camera_image', methods=["POST"])
def camera_image():
    global original_image, current_image
    encoded_data = request.form['form'].split(',')[-1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    current_image = copy.deepcopy(original_image)

    img = copy.deepcopy(original_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im0 = get_base64(img)
    im1 = get_base64(pencil_sketch_grey(copy.deepcopy(img)))
    im2 = get_base64(pencil_sketch_col(img))
    im3 = get_base64(HDR(img))
    im4 = get_base64(emboss_image(img))
    im5 = get_base64(median_blur(img))
    im6 = get_base64(grayscale(img))
    im7 = get_base64(invert(img))
    im8 = get_base64(Summer(img))
    im9 = get_base64(Winter(img))
    im10 = get_base64(gradient(img))
    im11 = get_base64(dialation(img))
    return jsonify({
        'status':{
            'original-image': str(im0),
            'pencil-sketch-grey':str(im1),
            'pencil-sketch-col':str(im2),
            'hdr':str(im3),
            'emboss-image':str(im4),
            'median-blur':str(im5),
            'grayscale':str(im6),
            'invert':str(im7),
            'summer':str(im8),
            'winter':str(im9),
            'gradient':str(im10),
            'dialation':str(im11)
        }
    })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global original_image, current_image
    # print(request)
    file = request.files['file']
    # print(file)
    image_bytes = file.getvalue()
    original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    current_image = copy.deepcopy(original_image)

    img = copy.deepcopy(original_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im0 = get_base64(img)
    im1 = get_base64(pencil_sketch_grey(copy.deepcopy(img)))
    im2 = get_base64(pencil_sketch_col(img))
    im3 = get_base64(HDR(img))
    im4 = get_base64(emboss_image(img))
    im5 = get_base64(median_blur(img))
    im6 = get_base64(grayscale(img))
    im7 = get_base64(invert(img))
    im8 = get_base64(Summer(img))
    im9 = get_base64(Winter(img))
    im10 = get_base64(gradient(img))
    im11 = get_base64(dialation(img))
    return jsonify({
        'status':{
            'original-image': str(im0),
            'pencil-sketch-grey':str(im1),
            'pencil-sketch-col':str(im2),
            'hdr':str(im3),
            'emboss-image':str(im4),
            'median-blur':str(im5),
            'grayscale':str(im6),
            'invert':str(im7),
            'summer':str(im8),
            'winter':str(im9),
            'gradient':str(im10),
            'dialation':str(im11)
        }
    })

# @app.route('/download_image', methods=['POST'])
# def download_image():
#     global current_image
#     img = copy.deepcopy(current_image)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img.astype("uint8"))
#     rawBytes = io.BytesIO()
#     img.save(rawBytes, "PNG")
#     rawBytes.seek(0)
#     img_base64 = base64.b64encode(rawBytes.read())
#     return jsonify({'status':str(img_base64)})

@app.route('/reset_slider', methods=['GET'])
def reset():
    global current_image, original_image
    img = copy.deepcopy(original_image)
    current_image = copy.deepcopy(original_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "PNG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

@app.route('/save_filter', methods=['GET'])
def save_current_filter():
    global current_image, original_image
    original_image = copy.deepcopy(current_image)
    return jsonify({'status':'ok'})

@app.route('/current_image', methods=['POST','GET'])
def current_image():
    global original_image, current_image, prev_filter_request    
    fail = 0

    filter_name = request.get_json()['filter']
    if request.get_json()['value']:
        value = float(request.get_json()['value'])

    img = copy.deepcopy(original_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print(value, filter_name)

    png = 0
    if filter_name == "brightness":
        # print(value)
        # img = bright(img, 0)
        img = bright(img, value)
    if filter_name == "contrast":
        value /= 100
        value += 1
        img = contrast(img, value)
    if filter_name == "hue":
        img = hue_shift(img, value)
    if filter_name == "saturation":
        img = saturation_shift(img, value)
    if filter_name == "value":
        img = value_shift(img, value)
    if filter_name == "blur":
        value = int(value)
        img = blur(img, value)
    if filter_name == "opacity":
        png = 1
        img = opacity(img, value)
    if filter_name == "crop-left":
        multiplier = 1-value
        img = crop(img, multiplier, 'left')
    if filter_name == "crop-right":
        multiplier = 1-value
        img = crop(img, multiplier, 'right')
    if filter_name == "crop-top":
        multiplier = 1-value
        img = crop(img, multiplier, 'top')
    if filter_name == "crop-bottom":
        multiplier = 1-value
        img = crop(img, multiplier, 'bottom')
    if filter_name == "resize-width":
        img = image_resize(img, value, 'width')
    if filter_name == "resize-height":
        img = image_resize(img, value, 'height')
    if filter_name == 'rotate':
        value = -value
        img = shear_rotate(img, value)
        img = np.float32(img)

    if filter_name == "seam-carving-crop":
        print(img.shape)
        img = seam_carv_crop(img)
        print(img.shape)
    if filter_name == "seam-carving-expand":
        print(img.shape)
        img = seam_carv_expand(img)
        print(img.shape)
    if filter_name == "dog-filter":
        img = dog_filter(img)
        if img == []:
            fail = 1
    if filter_name == "hat-filter":
        img = hat(img)
        if img == []:
            fail = 1
    if filter_name == "thug-filter":
        img = thug(img)
        if img == []:
            fail = 1
    if filter_name == 'remove-bg':
        img = bgremove2(img)

    if fail == 1:
        img = copy.deepcopy(original_image)

    prev_filter_request = filter_name
    
    


    current_image = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_RGB2BGR)
    

    #apply filter

    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    if png == 1:
        img.save(rawBytes, "PNG")
    else:
        img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    # print(img_base64)
    return jsonify({'status':str(img_base64), 'failure': fail})




@app.route('/main')
def main_page():
    return "MAIN PAGE"
