import csv
import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from DrawOnCamera import draw_landmarks, draw_bounding_rect, draw_claw_rect, draw_info_text, draw_info

import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int , default=1920)
    parser.add_argument("--height", help='cap height', type=int, default=1080)
    parser.add_argument('--use_static_image_mode', action='store_false')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.9)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)

    args = parser.parse_args()

    return args

def main():
    IP_ADDRESS_AND_PORT = "192.168.137.16:5000"
    args = get_args() #Gets - device(int), width(int), height(int), use_static_image_mode(string), min_detection_confidence(float), min_tracking_confidence(float)

    cap_device = args.device 
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = False
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    print(use_static_image_mode)
    use_brect = True #Used for toggling bounding Rectangle

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #  ########################################################################
    mode = 0
    repeat = 11 
    num = 0
    hand_mode = 10
    while True:
        num += 1
        fps = int(cvFpsCalc.get()) #Shows FPS

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10) #Adds Delay to input capture
        if key == 27:  # ESC
            break
        else:
            number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read() #Ret will be True of false, image is each frame
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                #MOVEMENT CONTROLS
                if (handedness.classification[0].label[0:] == "Left"):
                    if hand_sign_id==0:
                        if repeat != 0:
                            print("Brake")
                            # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/stop")
                        repeat = 0
                    elif hand_sign_id==1:
                        if repeat != 1:
                            print("Forward")
                            # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/front")
                        repeat = 1
                    elif hand_sign_id==2:
                        if repeat != 2:
                            print("Right")
                            # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/right")
                        repeat = 2
                    elif hand_sign_id==3:
                        if repeat != 3:
                            print("Left")
                            # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/left")
                        repeat = 3
                    elif hand_sign_id==4:
                        if repeat != 4:
                            print("Reverse")
                            # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/back")
                        repeat = 4

                #CLAW CONTROLS
                if (handedness.classification[0].label[0:] == "Right" ):
                    if ((landmark_list[9][0] >= (cap_width // 4)) and (landmark_list[9][0] <= ((cap_width // 5) * 4)) and (landmark_list[9][1] >= (cap_height // 4)) and (landmark_list[9][1] <= ((cap_height // 5) * 4)) ):
                        xpos = ((landmark_list[9][0] - (cap_width//2)) // 5)
                        ypos = (-(landmark_list[9][1] - (cap_height//2)) // 5 ) - 60
                        zpos = ((hand_landmarks.landmark[8].z * - 900) + 150)
                        #if (zpos > 150) and (zpos < 300) :
                        print(xpos , ypos , int(zpos))
                            #os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/arm?" + "xpos=" + str(xpos) + "^&" + "ypos=" + str(ypos) + "^&" + "zpos=" + str(int(zpos)))
                        if hand_sign_id==5 or hand_sign_id==7: 
                                #os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/claw_close")
                                print("Close")
                        if hand_sign_id==6: 
                                #os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/claw_open")
                                print("open")

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id])
        debug_image = draw_info(debug_image, fps, mode, number)
        debug_image = draw_claw_rect(True , debug_image , cap_width , cap_height)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


#This Function is used for adding new Keypoints
def select_mode(key,   mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (number != -1):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

if __name__ == '__main__':
    main()
