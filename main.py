import time
import cv2
import numpy as np
import pyautogui
import utils.HandDetector as hdm

cam_width, cam_height = 640*2, 480*2
screen_width, screen_height = pyautogui.size()
frameReductionWidth = 200
frameReductionHeight = 250

def get_landmarks_on_image(image, landmarks):
    h, w, _ = image.shape
    return np.array([(int(lm[0] * w), int(lm[1] * h), int(lm[2])) for lm in landmarks])

def smooth_movement(current, previous, alpha=0.3):
    return alpha * current + (1 - alpha) * previous

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)
    detector = hdm.HandDetector(maxHands=1)

    prev_x, prev_y = screen_width // 2, screen_height // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (cam_width, cam_height))
        frame = cv2.flip(frame, 1)
        hands, frame = detector.getHands(frame, draw=True)

        cv2.rectangle(frame, (frameReductionWidth, frameReductionHeight), (cam_width - frameReductionWidth, cam_height - frameReductionHeight), (0, 255, 0), 2)

        if len(hands) > 0:
            hand = hands[0]
            landmarks = hand.landmarks
            img_landmarks = get_landmarks_on_image(frame, landmarks)

            up_fingers = detector.getUpFingers(hand)
            index_finger = img_landmarks[hdm.FINGER_NAMES['Index']]
            middle_finger = img_landmarks[hdm.FINGER_NAMES['Middle']]

            x = np.interp(index_finger[0], (frameReductionWidth, cam_width - frameReductionHeight), (0, screen_width))
            y = np.interp(index_finger[1], (frameReductionWidth, cam_height - frameReductionHeight), (0, screen_height))

            x = smooth_movement(x, prev_x)
            y = smooth_movement(y, prev_y)

            x = max(0, min(screen_width - 1, x))
            y = max(0, min(screen_height - 1, y))
            
            prev_x, prev_y = x, y

            try:
                if up_fingers[1] == 1 and up_fingers[2] == 0:
                    cv2.circle(frame, index_finger[:2], 15, (0, 255, 0), -1)
                    pyautogui.moveTo(x, y)
                
                elif up_fingers[1] == 1 and up_fingers[2] == 1:
                    distance = detector.getDistance(index_finger, middle_finger)
                    if distance <= 30:
                        pyautogui.click()
                        time.sleep(0.1)
            except pyautogui.FailSafeException:
                prev_x, prev_y = x, y

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
