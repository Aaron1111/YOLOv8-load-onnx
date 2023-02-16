import cv2
# from imread_from_url import imread_from_url
from yolov8 import YOLOv8
from paddleocr import PaddleOCR
# Code to Measure time taken by program to execute.
import time
 
# store starting time
begin = time.time()


# Initialize yolov8 object detector
model_path = "runs/detect/yolov8n_v8_car_plate2/weights/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.0, iou_thres=0.0)

# init ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False) # need to run only once to load model into memory


# Read image
# img_url = "https://th.bing.com/th/id/OIP.KOYb9FCDzwrRgskgrtG6mwHaEK?pid=ImgDet&rs=1"
# img = imread_from_url(img_url)

## ganti path ini sesuai image yang ingin dideteksi
imgPath = "dataset/img_car_plate/img/19.jpg"


imgName = "carPlateONNX.jpg"
img = cv2.imread(imgPath)
# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

print("boxes", boxes)

def remove(string):
            return string.replace(" ", "")

for box in boxes:
    print("box:", box)
    print("x min: ", box[0])
    print("y min: ", box[1])
    print("x max: ", box[2])
    print("y max: ", box[3])
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = int(box[2])
    y_max = int(box[3])
    crop_img = img[y_min:y_max, x_min:x_max]
    cv2.imwrite("carPlateONNX.jpg",crop_img)
    result = ocr.ocr(imgName, det=False, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print("plate license: ", remove(line[0]))
            print("accuracy: ", line[1])


end = time.time()
# total time taken
print(f"Total runtime of the program is {end - begin} seconds")

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Plate License", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Plate License", combined_img)
# cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
 