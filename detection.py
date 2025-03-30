from ultralytics import YOLO
import cv2
import cvzone
import math
import PokerHandFunction

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("yolo-Weights/yolov8s_playing_cards.pt")
# model=YOLO("Playing-Cards.v3.yolov8/valid")
# model = YOLO("playingCards.pt")
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            if conf > 0.5:
                hand.append(classNames[cls])

    print(hand)
    hand = list(set(hand))
    print(hand)
    if len(hand) == 5:
        results = PokerHandFunction.findPokerHand(hand)
        print(results)
        cvzone.putTextRect(img, f'Your Hand: {results}', (300, 75), scale=3, thickness=5)

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2
# import math
# import PokerHandFunction
# # start webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# # model
# model = YOLO("yolo-Weights/playingCards.pt")
# # model = YOLO("yolo-Weights/yolov8s_playing_cards.pt")
#
# # object classes
# classNames = [ "10C", "10D", "10H", "10S",
#              "2C", "2D", "2H", "2S",
#              "3C", "3D", '3H', '3S',
#              '4C', '4D', '4H', '4S',
#              '5C', '5D', '5H', '5S',
#              '6C', '6D', '6H', '6S',
#              '7C', '7D', '7H', '7S',
#              '8C', '8D', '8H', '8S',
#              'QC', 'QD', 'QH', 'QS',
#              '9C', '9D', '9H', '9S',
#              'AC', 'AD', 'AH', 'AS',
#              'JC', 'JD', 'JH', 'JS',
#              'KC', 'KD', 'KH', 'KS'
#               ]
#
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#
#     # coordinates
#     for r in results:
#         boxes = r.boxes
#
#         for box in boxes:
#             # bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
#
#             # put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#             # confidence
#             confidence = math.ceil((box.conf[0]*100))/100
#             print("Confidence --->",confidence)
#
#             # class name
#             cls = int(box.cls[0])
#             print("Class name -->", classNames[cls])
#
#             # object details
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2
#
#             cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#             if confidence > 0.5:
#                 hand.append(classNames[cls])
#
#                 print(hand)
#             hand = list(set(hand))
#             print(hand)
#             if len(hand) == 5:
#                 results = PokerHandFunction.findPokerHand(hand)
#                 print(results)
#
#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush",
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import PokerHandFunction
#
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
#
#
# model = YOLO("yolo-Weights/playingCards.pt")
# # model = YOLO("playingCards.pt")
# classNames = ['10C', '10D', '10H', '10S',
#               '2C', '2D', '2H', '2S',
#               '3C', '3D', '3H', '3S',
#               '4C', '4D', '4H', '4S',
#               '5C', '5D', '5H', '5S',
#               '6C', '6D', '6H', '6S',
#               '7C', '7D', '7H', '7S',
#               '8C', '8D', '8H', '8S',
#               '9C', '9D', '9H', '9S',
#               'AC', 'AD', 'AH', 'AS',
#               'JC', 'JD', 'JH', 'JS',
#               'KC', 'KD', 'KH', 'KS',
#               'QC', 'QD', 'QH', 'QS']
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     hand = []
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#             if conf > 0.5:
#                 hand.append(classNames[cls])
#
#     print(hand)
#     hand = list(set(hand))
#     print(hand)
#     if len(hand) == 5:
#         results = PokerHandFunction.findPokerHand(hand)
#         print(results)
#         cvzone.putTextRect(img, f'Your Hand: {results}', (300, 75), scale=3, thickness=5)
#
#     cv2.imshow("Webcam", img)
#     cv2.waitKey(1)

# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
#
# # cap = cv2.VideoCapture(1)  # For Webcam
# # cap.set(3, 1280)
# # cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/yolov8n.mp4")  # For Video
#
# model = YOLO("yolo-Weights/yolov8n.pt")
#
# classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#               'Safety Vest', 'machinery', 'vehicle']
# myColor = (0, 0, 255)
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             # cvzone.cornerRect(img, (x1, y1, w, h))
#
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#             currentClass = classNames[cls]
#             print(currentClass)
#             if conf > 0.5:
#                 if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
#                     myColor = (0, 0, 255)
#                 elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
#                     myColor = (0, 255, 0)
#                 else:
#                     myColor = (255, 0, 0)
#
#                 cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
#                                    (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
#                                    colorT=(255, 255, 255), colorR=myColor, offset=5)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
# same same but different
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import PokerHandFunction
#
# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
# model = YOLO("yolo-Weights/yolov8s_playing_cards.pt")
# # model=YOLO("Playing-Cards.v3.yolov8/valid")
# # model = YOLO("playingCards.pt")
# classNames = ['10C', '10D', '10H', '10S',
#               '2C', '2D', '2H', '2S',
#               '3C', '3D', '3H', '3S',
#               '4C', '4D', '4H', '4S',
#               '5C', '5D', '5H', '5S',
#               '6C', '6D', '6H', '6S',
#               '7C', '7D', '7H', '7S',
#               '8C', '8D', '8H', '8S',
#               '9C', '9D', '9H', '9S',
#               'AC', 'AD', 'AH', 'AS',
#               'JC', 'JD', 'JH', 'JS',
#               'KC', 'KD', 'KH', 'KS',
#               'QC', 'QD', 'QH', 'QS']
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     hand = []
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#             if conf > 0.5:
#                 hand.append(classNames[cls])
#
#     print(hand)
#     hand = list(set(hand))
#     print(hand)
#     if len(hand) == 5:
#         results = PokerHandFunction.findPokerHand(hand)
#         print(results)
#         cvzone.putTextRect(img, f'Your Hand: {results}', (300, 75), scale=3, thickness=5)
#
#     # cv2.imshow("Image", img)
#     # cv2.waitKey(1)
#     cv2.imshow('Webcam',img)
#     cv2.waitKey(1)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()