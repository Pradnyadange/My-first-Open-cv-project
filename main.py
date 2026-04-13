import cv2
import mediapipe as mp
import numpy as np
import random
import math
import threading
from playsound import playsound

WIN_W, WIN_H = 640, 360
SPAWN_RATE = 35

BOOK_IMG = "assets/book.png"
INSTA_IMG = "assets/instagram.png"

YAY_SOUND = "assets/yayy.mp3"
def play_sound(path):
    def run():
        try:
            playsound(path)
        except:
            pass
    threading.Thread(target=run, daemon=True).start()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

def get_finger(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark[8]
        return int(lm.x * WIN_W), int(lm.y * WIN_H)
    return None, None
class Icon:
    def __init__(self):
        self.type = random.choice(["book", "insta"])
        self.img = cv2.imread(BOOK_IMG if self.type == "book" else INSTA_IMG, cv2.IMREAD_UNCHANGED)

        self.x = random.randint(60, WIN_W - 60)
        self.y = WIN_H + 50
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-12, -16)
        self.size = 70

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.25

    def draw(self, frame):
        if self.img is None:
            return

        img = cv2.resize(self.img, (self.size, self.size))
        h, w = img.shape[:2]

        x1 = int(self.x - w // 2)
        y1 = int(self.y - h // 2)
        x2 = x1 + w
        y2 = y1 + h

        if x2 < 0 or y2 < 0 or x1 > WIN_W or y1 > WIN_H:
            return

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(WIN_W, x2), min(WIN_H, y2)

        roi = frame[y1c:y2c, x1c:x2c]

        img_crop = img[
            y1c - y1:y2c - y1,
            x1c - x1:x2c - x1
        ]

        if img_crop.shape[2] == 4:
            alpha = img_crop[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            roi[:] = roi * (1 - alpha) + img_crop[:, :, :3] * alpha
        else:
            roi[:] = img_crop
icons = []
book_score = 0
insta_score = 0
frame_count = 0
game_over = False

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIN_W, WIN_H))

    fx, fy = get_finger(frame)
    if not game_over:
        frame_count += 1
        if frame_count % SPAWN_RATE == 0:
            icons.append(Icon())

    for icon in icons[:]:
        icon.update()

        if fx is not None:
            dist = math.hypot(icon.x - fx, icon.y - fy)

            if dist < 40:
                if icon.type == "book":
                    book_score += 1
                    play_sound(YAY_SOUND)
                else:
                    insta_score += 1

                icons.remove(icon)

        if icon.y < -80:
            icons.remove(icon)
    for icon in icons:
        icon.draw(frame)

    if fx is not None:
        cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

    cv2.putText(frame, f"Book Score: {book_score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Insta Score: {insta_score}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,100,100), 2)

    if insta_score > book_score:
        game_over = True

    if game_over:
        cv2.putText(frame, "PADHLE SAALE..", (90, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

        cv2.putText(frame, "Press R to Restart or Q to Quit", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        key = cv2.waitKey(1)

        if key == ord('r'):
            icons = []
            book_score = 0
            insta_score = 0
            game_over = False
            frame_count = 0

        if key == ord('q'):
            break

    cv2.imshow("Icon Ninja", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()