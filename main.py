import cv2
import numpy as np
import argparse
LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
video = "parking.mp4"
CONFIDENCE_THRESHOLD = 0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = "uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

def drawBoxes (image, layerOutputs, H, W):
  boxes = []
  confidences = []
  classIDs = []

  for output in layerOutputs:
    for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

      if confidence > CONFIDENCE_THRESHOLD:
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)

  idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

  # Ensure at least one detection exists
  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      color = [int(c) for c in COLORS[classIDs[i]]]

      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Display the image
  cv2.imshow("output", image)

def detectObjects(imagePath):
      image = cv2.imread(imagePath)
      (H, W) = image.shape[:2]

      ln = net.getLayerNames()
      ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

      blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
      net.setInput(blob)
      layerOutputs = net.forward(ln)
      drawBoxes(image, layerOutputs, H, W)



if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True, help = "Path to input file")

  args = vars(ap.parse_args())
  detectObjects(args["image"])

  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Ниже расположен код для считывания видео
# Также стоит заметить, что скорее всего получение сетки довольно бессмысленно так как сеть нормально определяет авто
# Далее требуется реализовать функцию распознавания машин на видео. В качестве видео сделана миниатюра парковки.
# Для распознования машин можно проверять не каждый кадр, а хотя бы каждый 10 для снижения нагрузки на оборудование.
# К сожалению, сеть не может определить машину на фото. Поэтому требуется сделать видео с нормальной парковки.
# После реализации обнаружения объекта на видео требуется сделать счетчик количества машин находящихся на парковке
# 1 вариант - общий подсчет количества машин. 2 вариант - подсчет машин, находящихся между разметкой.


# запуск программы -  python main.py -i *.jpg

# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# labels = open("coco.names").read().strip().split("\n")
# video = 'parking.mp4'
# cap = cv2.VideoCapture(video)
# while cap.isOpened():
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# def YOLO_algorithm():
#