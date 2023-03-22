from time import time
import torch
import numpy as np
import cv2
import mediapipe as mp
import imutils

classnames = ['T-shirt', 'LongSleeve', 'Trousers']

def Clothes_Model_Loader(weight_path='best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path, force_reload=True)
    return model

def frame_loader(frame, imsize):
    """
    processes input frame for inference
    """
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (imsize, imsize))
    frame = frame[:, :, ::-1].transpose(2, 0, 1)
    frame = np.ascontiguousarray(frame)
    frame = torch.from_numpy(frame)
    frame = frame.float()
    frame /= 255.0
    frame = frame.unsqueeze(0)
    return frame, h, w

def get_pred_results(model):
    confidence = clothes_model.pandas().xyxy[0]['confidence'].values
    detected_class = clothes_model.pandas().xyxy[0]['name'].values
    print(f"Confidence: {confidence}")
    print(f"Detected classes: {detected_class}")


"""
Hand Detection Part
"""
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

def draw_bounding_box(img, results):
    """
    Args:
        img: <class 'numpy.ndarray'>
        results:

    Returns:
    """
    if results.multi_hand_landmarks:
        for hand_landmark, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):
            img_height, img_width, _ = img.shape
            x = [int(landmark.x * img_width) for landmark in hand_landmark.landmark]
            y = [int(landmark.y * img_height) for landmark in hand_landmark.landmark]
            score = np.mean([float(classification.score) for classification in hand_classification.classification])
            score = "{:.2f}".format(round(score, 2))

            left = np.min(x)
            right = np.max(x)
            bottom = np.min(y)
            top = np.max(y)

            thick = int((img_height + img_width) // 400)

            line_width = max(round(sum(img.shape) / 2 * 0.003), 2)  # line width

            # Bouding box visualization
            cv2.rectangle(img,
                          (left - 10, top + 10),    # Top left coordinates
                          (right + 10, bottom - 10),    # Bottom right coordinates
                          (255, 0, 0),  # Color of the detection box
                          thickness=line_width,
                          lineType=cv2.LINE_AA)

            # Text info display on bounding box
            tf = max(line_width - 1, 1)  # font thickness

            # text width, height
            w, h = cv2.getTextSize(f'Hand {score}', 0, fontScale=line_width / 3, thickness=tf)[0]
            outside = (left - 10) - h >= 3
            p2 = (left - 10) + w, (top + 10) - h - 3 if outside else (top + 10) + h + 3
            cv2.rectangle(img, (left - 10, top + 10), p2, (255, 0, 0), -1, cv2.LINE_AA)  # filled
            cv2.putText(img,
                        f'Hand {score}', ((left - 10), (top + 10) - 2 if outside else (top + 10) + h + 2),
                        0,
                        line_width / 3,
                        (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

def get_countours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    result = gray_img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("x,y,w,h:", x, y, w, h)

def is_hand_detected(results):
    if results.multi_hand_landmarks and results.multi_handedness:
        print("Hands Detected! Stop Folding")
        return True
    else:
        print("Keep Folding")
        return False


if __name__ == '__main__':
    # Path of pretrained weight for clothes model
    weight_path = 'Merged_Exp3/weights/best.pt'
    # Load model
    model = Clothes_Model_Loader(weight_path=weight_path)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start = time()

        ret, frame, = cap.read()

        # For clothes detection
        clothes_model = model(frame)
        get_pred_results(clothes_model)

        # For hands detection
        frame = imutils.resize(frame, width=640, height=640)
        results = process_image(frame)
        draw_bounding_box(frame, results)
        is_hand_detected(results)

        parallel = np.concatenate((frame, np.squeeze(clothes_model.render())), axis=0)

        cv2.imshow('Parallel', parallel)

        # cv2.imshow("Hand Detection", frame)
        # cv2.imshow('Clothes Detection', np.squeeze(clothes_model.render()))


        # if cv2.waitKey(10) & 0xff == ord('x'):
        #     break
        # if cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE) < 1:
        #     break

        end = time()
        fps = 1 / (end - start)
        # print(fps)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

    # cap.release()
    # cv2.destroyAllWindows()
