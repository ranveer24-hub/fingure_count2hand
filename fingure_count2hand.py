import cv2
import mediapipe as mp

# --------------------------
# MediaPipe Hands init
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dono hands detect karne ke liye max_num_hands=2
hands = mp_hands.Hands(max_num_hands=2)

# Finger tip landmark IDs (MediaPipe numbering)
# Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
tip_ids = [4, 8, 12, 16, 20]

# --------------------------
# Webcam start
# --------------------------
cap = cv2.VideoCapture(0)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Selfie view (mirror) - isse display natural lagta hai
    img = cv2.flip(img, 1)

    # MediaPipe ko RGB chahiye
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0  # dono hands ka sum

    if results.multi_hand_landmarks:
        # IMPORTANT: handedness ko har hand ke saath zip karo
        # taa ki "Left/Right" label sahi hand pe lage
        for hand_landmarks, handed in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
            h, w, _ = img.shape
            lm_list = []

            # Sare landmarks ke pixel coords banao
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []  # current hand ki state list

            # ------------------------------
            # Thumb (angutha) check
            # ------------------------------
            # Logic: Right hand -> tip.x > ip.x  (open)
            #        Left  hand -> tip.x < ip.x  (open)
            label = handed.classification[0].label  # "Left" ya "Right"
            thumb_tip_x = lm_list[tip_ids[0]][0]      # 4
            thumb_ip_x  = lm_list[tip_ids[0] - 1][0]  # 3

            if label == "Right":
                thumb_open = thumb_tip_x < thumb_ip_x
            else:  # "Left"
                thumb_open = thumb_tip_x > thumb_ip_x

            fingers.append(1 if thumb_open else 0)

            # ------------------------------
            # Baaki 4 fingers (Index→Pinky)
            # Tip.y < PIP.y  ==> finger open (kyunki top side pe hota hai)
            # IDs: tip -> 8/12/16/20, PIP -> 6/10/14/18  (tip-2)
            for i in range(1, 5):
                tip_y = lm_list[tip_ids[i]][1]
                pip_y = lm_list[tip_ids[i] - 2][1]
                fingers.append(1 if tip_y < pip_y else 0)

            # Is hand ka count add karo total me
            total_fingers += fingers.count(1)

            # Landmarks draw
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # OPTIONAL: Hand label dikhane ke liye (debug/helpful)
            cv2.putText(img, label, (lm_list[0][0] + 10, lm_list[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Total count screen par
    cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("FingerTrack AI - Both Hands", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
