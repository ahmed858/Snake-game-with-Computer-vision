from black import main
import cv2
import math
import cvzone
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector


class Snake_game:
    def __init__(self, image_dir):
        self.points = []
        self.distances = []
        self.cur_len = 0
        self.allowed_len = 50
        self.prev_point = 0, 0

        self.food_img = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)
        self.food_h, self.food_w, _ = self.food_img.shape
        self.food_point = 0, 0
        self.random_location()
        self.score = 0

        self.game_over = False

    def random_location(self):
        self.food_point = random.randint(100, 900), random.randint(100, 500)

    def update(self, main_Img, cur_head):

        if self.game_over:
            cvzone.putTextRect(
                main_Img, "GAME OVER!! ", [300, 400], scale=5, thickness=3, offset=30
            )
            cvzone.putTextRect(
                main_Img,
                "Score {}".format(self.score),
                [300, 500],
                scale=5,
                thickness=3,
                offset=30,
            )

        else:
            cur_x, cur_y = cur_head
            prev_x, prev_y = self.prev_point

            self.points.append([cur_x, cur_y])
            dis = math.hypot(cur_x - prev_x, cur_y - prev_y)
            self.distances.append(dis)
            self.cur_len += dis
            self.prev_point = cur_x, cur_y

            # length reduction
            if self.cur_len > self.allowed_len:
                for i, length in enumerate(self.distances):
                    self.cur_len -= length
                    self.points.pop(i)
                    self.distances.pop(i)

                    if self.cur_len < self.allowed_len:
                        break

            # check snake ate the food
            x, y = self.food_point
            if (
                x - self.food_w // 2 < cur_x < x + self.food_w // 2
                and y - self.food_h // 2 < cur_y < y + self.food_h // 2
            ):
                self.random_location()
                self.allowed_len += 50
                self.score += 5

            # Drow the snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(
                            main_Img,
                            self.points[i - 1],
                            self.points[i],
                            (0, 255, 0),
                            15,
                        )

                cv2.circle(main_Img, self.points[-1], 20, (0, 0, 255), 5)

            # drow food
            main_Img = cvzone.overlayPNG(
                main_Img, self.food_img, (x - self.food_w // 2, y - self.food_h // 2)
            )

            cvzone.putTextRect(
                main_Img,
                "Score {}".format(self.score),
                [100, 50],
                scale=3,
                thickness=3,
                offset=10,
            )
            # check for collistion
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(main_Img, [pts], False, (200, 0, 0), 3)
            mn_dis = cv2.pointPolygonTest(pts, (cur_x, cur_y), True)

            if -1 <= mn_dis <= 1:
                print("Hit")
                self.game_over = True
                self.points = []
                self.distances = []
                self.cur_len = 0
                self.allowed_len = 50
                self.prev_point = 0, 0
                self.random_location()
                self.score = 0

            return main_Img


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    hand_detector = HandDetector(detectionCon=0.8, maxHands=1)

    game = Snake_game(
        "D:\VS Code\Pythonesta\DataSceince\Computer Vision\Snake game with Computer vision\Donut.png"
    )

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1100, 700))

        img = cv2.flip(img, 1)
        hands, img = hand_detector.findHands(img, flipType=False)

        if hands:
            lmList = hands[0]["lmList"]
            headIndex = lmList[8][0:2]  # just x,y
            img = game.update(img, headIndex)

        cv2.imshow("Game", img)
        if cv2.waitKey(30) == ord("1"):
            break
