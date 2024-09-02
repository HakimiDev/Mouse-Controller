import time
import mediapipe as mp
from google.protobuf.json_format import MessageToDict 
import cv2
import numpy as np

FINGER_NAMES = {
    'Thumb': 4,
    'Index': 8,
    'Middle': 12,
    'Ring': 16,
    'Pinky': 20
}

class Hand:
    def __init__(self, landmarks, bbox, hand_type):
        self.landmarks = np.array(landmarks)
        self.bbox = np.array(bbox)
        self.type = hand_type

    def isFlipped(self):
        wrist = self.landmarks[0]
        thumb_tip = self.landmarks[FINGER_NAMES['Thumb']]
        is_flipped = thumb_tip[0] > wrist[0]

        return is_flipped if self.type == 'Right' else not is_flipped

class HandDetector(mp.solutions.hands.Hands):
    def __init__(self, mode=False, dConfidence=0.7, tConfidence=0.7, maxHands=2, mCmplexity=1):
        super().__init__(mode, maxHands, mCmplexity, dConfidence, tConfidence)
        self.drawing = mp.solutions.drawing_utils

    def getHands(self, image, draw=False):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.process(rgb_image)
        hands_list = []

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for i, landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
                x_coords, y_coords = landmarks_array[:, 0], landmarks_array[:, 1]

                bbox = np.array([
                    int(np.min(x_coords) * w),
                    int(np.max(x_coords) * w),
                    int(np.min(y_coords) * h),
                    int(np.max(y_coords) * h)
                ])

                hand_type = MessageToDict(results.multi_handedness[i])['classification'][0]['label'] 
                hand = Hand(
                    landmarks=landmarks_array,
                    bbox=bbox,
                    hand_type=hand_type
                )
                hands_list.append(hand)

                if draw:
                    self.drawing.draw_landmarks(
                        image, landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.drawing.DrawingSpec(color=(0, 0, 255), circle_radius=5, thickness=-1),
                        self.drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
                    )

        return np.array(hands_list), image
    
    def getUpFingers(self, hand):
        landmarks = hand.landmarks
        hand_type = hand.type
        is_flipped = hand.isFlipped()

        up_fingers = []
        for finger_name, tip_index  in FINGER_NAMES.items():
            tip = landmarks[tip_index]
            base = landmarks[tip_index - 1] if finger_name == 'Thumb' else landmarks[tip_index - 2]

            if finger_name == 'Thumb':
                if hand_type == 'Right':
                    is_up = tip[0] < base[0]
                    up_fingers.append(is_up if not is_flipped else not is_up)
                else:
                    is_up = tip[0] > base[0]
                    up_fingers.append(is_up if not is_flipped else not is_up)
            else:
                is_up = tip[1] < base[1]
                up_fingers.append(is_up)

        return np.array(up_fingers)
    
    def getDistance(self, point1, point2):
        x1, y1, _ = point1
        x2, y2, _ = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hands, frame = detector.getHands(frame, draw=True)
        # if len(hands) > 1:
        #     point1 = hands[0].landmarks[FINGER_NAMES['Index']]
        #     point2 = hands[1].landmarks[FINGER_NAMES['Index']]
        #     print(detector.getDistance(point1, point2))

        for hand in hands:
            type = hand.type
            bbox = hand.bbox
            frame = cv2.putText(frame, type, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            frame = cv2.putText(frame, "Flipped" if hand.isFlipped() else "Not Flipped", (bbox[0], bbox[2] - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (25, 255, 32), 2)
            # print(detector.getUpFingers(hand))

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        frame = cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
