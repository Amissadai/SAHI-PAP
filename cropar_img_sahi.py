from random import randrange
from uuid import uuid4
import cv2


frame = cv2.imread('2.jpg')
frame = cv2.resize(frame, (1280, 640), interpolation=cv2.INTER_AREA)

copy = frame.copy()

print(frame.shape)

for i in range(33): # 33
    for j in range(88): # 88
        copy = frame.copy()
        x = j + (10*j)
        y = i + (10*i)
        x2 = x + 320
        y2 = 320 + (10*i)

        cv2.rectangle(copy, (x,y), (x2,y2), (0,0,255), 3)
        cv2.imshow("frame", copy)

        name = uuid4()
        #cv2.imwrite(f'crops/{name}.png', frame[y:y2, x:x2])
        cv2.waitKey(33)