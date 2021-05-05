import glob
import cv2
import pytesseract
import argparse
import utils
utils.remove_line_images()


# Argument Parsing
# --image /path/to/image
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="path to image")
args = parser.parse_args()
image_path = args.image


# Plate Detection
# Save as [cropped_image.jpg]
print("Detecting Plate...")
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)

image = cv2.imread(image_path)
classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

for idx, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
    print(box, score)
    x, y, w, h = box
    roi_box = image[y: y+h, x: x+w]
    cv2.imwrite("cropped_image.jpg", roi_box)


# Correcting Skewness
# Save as [rotated_image.jpg]
image = cv2.imread("cropped_image.jpg")
angle, rotated = utils.correct_skew(image)
cv2.imwrite("rotated_image.jpg", rotated)


# ------------------------------------------------------------------
# RC Starts


# Line Detection
print("Detecting Lines...")
image_path = "rotated_image.jpg"

net = cv2.dnn.readNet("yolo_line.weights", "yolo_line.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(128, 128), scale=1/255, swapRB=True)

image = cv2.imread(image_path)
classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

linedict = dict()
for idx, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
    print(box, score)
    x, y, w, h = box
    roi_box = image[y: y+h, x: x+w]
    linedict[y] = roi_box
#    cv2.imwrite(f"line{idx}.jpg", roi_box)

utils.save_lines(linedict)


# Recognizing lines
# TODO Sweep parameters

print("Recognizing...")

lines = sorted(glob.glob("line*.jpg"))

tesseract_text = ""
for line in lines:
    image = cv2.imread(line)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#    blur = cv2.GaussianBlur(gray, (3, 3), 0)

#TODO gray to blur
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

#    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#    erosion = cv2.erode(thresh, rect_kern, iterations=2)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    dilation = cv2.bitwise_not(dilation)

    config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13 --oem 1 --tessdata-dir ./tessdata'
    text = pytesseract.image_to_string(dilation, lang='foo', config=config)
    text = utils.remove_alnum(text)
    tesseract_text += text
    print(text)

utils.save_tesseract_text(tesseract_text)
print(tesseract_text)


utils.clean()
