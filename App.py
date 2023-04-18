import csv
import copy
import argparse
import itertools
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc, select_mode, Drawings

from model import KeyPointClassifier

import os

#Argument Parsing
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int , default=1920)
    parser.add_argument("--height", help='cap height', type=int, default=1080)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    cvFpsCalc = CvFpsCalc(buffer_len=1)

    IP_ADDRESS_AND_PORT = '172.20.10.7' #Change When switching wifi on robot

    #Camera Setup
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #Mediapipe Model Loading
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    mode = 0
    Lrepeat = None
    Rrepeat = None
    delay = 0
    while True:
        delay += 1
        fps = int(cvFpsCalc.get())
        key = cv.waitKey(10)
        if key == 27: #ESC key is pressed
            break
        else:
            number, mode = select_mode.select_mode(key, mode)
        
        #Camera Capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        debug_image = Drawings.draw_info(debug_image, fps, mode, number)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                b_rect = calc_bounding_rect(debug_image, hand_landmarks)
                
                landmark_list = calc_landmark_list(debug_image,hand_landmarks)
                
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                logging_csv(number,mode,pre_processed_landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                #Here is where I will add the controls to the robot
                if (handedness.classification[0].label[0:] == "Left"):
                    if hand_sign_id==0:
                        if Lrepeat != 0:
                            print("Brake")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/stop")
                        Lrepeat = 0
                    elif hand_sign_id==1:
                        if Lrepeat != 1:
                            print("Forward")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/front")
                        Lrepeat = 1
                    elif hand_sign_id==2:
                        if Lrepeat != 2:
                            print("Right")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/right")
                        Lrepeat = 2
                    elif hand_sign_id==3:
                        if Lrepeat != 3:
                            print("Left")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/left")
                        Lrepeat = 3
                    elif hand_sign_id==4:
                        if Lrepeat != 4:
                            print("Reverse")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/back")
                        Lrepeat = 4
                    elif hand_sign_id==7:
                        if Lrepeat != 7:
                            print("Donut")
                            os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/donut")
                        Lrepeat = 7



                #CLAW CONTROLS
                if (handedness.classification[0].label[0:] == "Right" ):
                    if ((landmark_list[9][0] >= (cap_width // 4)) and (landmark_list[9][0] <= ((cap_width // 5) * 4)) and (landmark_list[9][1] >= (cap_height // 4)) and (landmark_list[9][1] <= ((cap_height // 5) * 4)) ):
                        xpos = ((landmark_list[9][0] - (cap_width//2)) // 5)
                        ypos = (-(landmark_list[9][1] - (cap_height//2)) // 5 ) - 60
                        zpos = ((hand_landmarks.landmark[8].z * - 900) + 150) 
                        # if (zpos > 150) and (zpos < 300):
                        if hand_sign_id==5: 
                                if delay%5 == 0:
                                    print(xpos , ypos , int(zpos))
                                    os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/arm?" + "xpos=" + str(xpos) + "^&" + "ypos=" + str(ypos) + "^&" + "zpos=" + str(int(zpos)))
                                if Rrepeat != 5:
                                    os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/claw_close")
                                    print("Close")
                                    Rrepeat = 5
                        if hand_sign_id==6:
                                if delay%5 == 0:
                                    print(xpos , ypos , int(zpos))
                                    os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/arm?" + "xpos=" + str(xpos) + "^&" + "ypos=" + str(ypos) + "^&" + "zpos=" + str(int(zpos)))
                                if Rrepeat!= 6: 
                                    os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/claw_open")
                                    print("open")
                                    Rrepeat = 6
                        if hand_sign_id==8:
                            if Rrepeat != 8:
                                print("StandTall")
                                os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/standtall")
                            Rrepeat = 8

                            # if hand_sign_id==9:
                            #     print("Dance")
                                # os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/dance")
                                # time.sleep(8)
                            # if hand_sign_id==9:
                            #     if Rrepeat != 9:
                            #         print("Clap")
                            #         os.popen("curl http://" + IP_ADDRESS_AND_PORT + "/dance")
                            #         time.sleep(8)
                            #     Rrepeat = 9

                #Drawings to display
                debug_image = Drawings.draw_bounding_rect(True, debug_image, b_rect)
                debug_image = Drawings.draw_landmarks(debug_image, landmark_list)
                debug_image = Drawings.draw_info_text(debug_image, b_rect, handedness, keypoint_classifier_labels[hand_sign_id])
                # debug_image = Drawings.draw_claw_rect(True , debug_image , cap_width , cap_height)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


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
