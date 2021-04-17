import cv2
import argparse
import easyocr
import keras_ocr
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--image', help="path to image")
args = parser.parse_args()
image_path = args.image


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




# Without Skewness outputs
easyreader = easyocr.Reader(['en'])
easyresult = easyreader.readtext("cropped_image.jpg")
easywiskew = [] 
for e in easyresult:
    easywiskew.append(e[1].upper())

pipeline = keras_ocr.pipeline.Pipeline()
images = ['cropped_image.jpg']
prediction = pipeline.recognize(images)[0]
keraswiskew = []
for p in prediction:
    keraswiskew.append(p[0].upper())



# Correcting Skewness
image = cv2.imread("cropped_image.jpg")
angle, rotated = utils.correct_skew(image)
cv2.imwrite("rotated_image.jpg", rotated)



# With Skewness outputs
easyreader = easyocr.Reader(['en'])
easyresult = easyreader.readtext("rotated_image.jpg")
easynoskew = [] 
for e in easyresult:
    easynoskew.append(e[1].upper())

pipeline = keras_ocr.pipeline.Pipeline()
images = ["rotated_image.jpg"]
prediction = pipeline.recognize(images)[0]
kerasnoskew = []
for p in prediction:
    kerasnoskew.append(p[0].upper())

easywiskew = utils.remove_alnum_from_list(easywiskew)
easynoskew = utils.remove_alnum_from_list(easynoskew)
keraswiskew = utils.remove_alnum_from_list(keraswiskew)
kerasnoskew = utils.remove_alnum_from_list(kerasnoskew)

print("-"*20)
print("EasyOCR Wi Skew", easywiskew)
print("EasyOCR No Skew", easynoskew)
print("KerasOCR Wi Skew", keraswiskew)
print("KerasOCR No Skew", kerasnoskew)

utils.save_easyocr_text(easywiskew, easynoskew)
utils.save_kerasocr_text(keraswiskew, kerasnoskew)


utils.clean()
