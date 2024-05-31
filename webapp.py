import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from base64 import b64encode
import matplotlib
from collections import defaultdict
import json
import webcolors

model = YOLO('best.pt')
UPLOAD_FOLDER = 'upload_folder'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

cumulative_count = 0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global cumulative_count

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if 'webcam_image_data' in request.files:

            file = request.files['webcam_image_data']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)


            img = cv2.imread(file_path)

            if img is None:
                flash('Error: Failed to load the image')
                return redirect(request.url)

            output_img, classes, colorHex = inference_img(img)
            _, buffer = cv2.imencode('.jpg', output_img)
            b64_img = b64encode(buffer).decode()

            cumulative_count += classes["Total Corn"]

            return render_template('result.html', b64_img=b64_img, classes=classes, cumulative_count=cumulative_count)

    return render_template('upload.html')

def inference_img(img):
    imgsize = 1280
    results = model.predict(img, imgsz=imgsize)
    output_img = img.copy()
    class_maps = [
        "Total Corn",
    ]
    colorsRGB = matplotlib.cm.tab20(range(len(class_maps)))
    colors = [(i[:-1][::-1] * 255) for i in colorsRGB]
    colorsRev = [(i[:-1][::1] * 255) for i in colorsRGB]
    colorsTuple = [(int(x), int(y), int(z)) for x, y, z in colorsRev]
    colorHex = {x: webcolors.rgb_to_hex(y) for x, y in zip(class_maps, colorsTuple)}
    classes_found = defaultdict(int)
    for result in results:
        boxes = result.boxes.to('cpu').numpy()
        classes = boxes.cls.astype(int)
        for box, cls in zip(boxes, classes):
            bbox_class = class_maps[cls]
            coord = box.xyxy.astype(int).squeeze()
            xmin, ymin, xmax, ymax = coord
            classes_found[bbox_class] += 1
            color = colors[cls]
            color = tuple(color)
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
    return output_img, classes_found, colorHex

@app.route('/reset', methods=['POST'])
def reset_count():
    global cumulative_count
    cumulative_count = 0
    return redirect('/')
@app.route('/capture_webcam', methods=['POST'])
def capture_webcam(camera=None):
    global cumulative_count

    try:

        if 'camera' in globals():
            camera.release()


        camera = cv2.VideoCapture(0)


        if not camera.isOpened():
            print("Error: Could not open webcam")
            return redirect('/')


        desired_exposure_value = 0.1


        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        camera.set(cv2.CAP_PROP_EXPOSURE, 0.2 )

        ret, frame = camera.read()

        if ret:
            filename = 'webcam_capture.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(file_path, frame)


            camera.release()

            img = cv2.imread(file_path)
            output_img, classes, colorHex = inference_img(img)
            _, buffer = cv2.imencode('.jpg', output_img)
            b64_img = b64encode(buffer).decode()

            cumulative_count += classes["Total Corn"]

            return render_template('result.html', b64_img=b64_img, classes=classes, cumulative_count=cumulative_count)

    except Exception as e:
        print("Error capturing from webcam:", str(e))

    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
