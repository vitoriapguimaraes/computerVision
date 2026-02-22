import cv2

try:
    import google.protobuf.internal
    import google.protobuf.internal.builder

    google.protobuf.internal.builder = google.protobuf.internal.builder
except ImportError:
    pass
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw


class HandTracker:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:

            self.mp_hands = mp_hands
            self.mp_draw = mp_draw
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                my_hand = {}
                mylmList = []
                for id, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])

                my_hand["lmList"] = mylmList
                my_hand["type"] = hand_type.classification[0].label
                all_hands.append(my_hand)

                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )

        return img, all_hands

    def fingers_up(self, my_hand):
        my_hand_type = my_hand["type"]
        my_lm_list = my_hand["lmList"]

        if len(my_lm_list) != 21:
            return []

        fingers = []
        # Thumb
        if my_hand_type == "Right":
            fingers.append(1 if my_lm_list[4][0] < my_lm_list[3][0] else 0)
        else:
            fingers.append(1 if my_lm_list[4][0] > my_lm_list[3][0] else 0)

        # 4 Fingers
        for id in range(8, 21, 4):
            fingers.append(1 if my_lm_list[id][1] < my_lm_list[id - 2][1] else 0)

        return fingers
