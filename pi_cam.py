import cv2
import numpy as np
import math
import os
from keras.models import load_model

classes = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16,
    'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24,
    'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32,
    'x': 33, 'y': 34, 'z': 35,
}

model = load_model('asl_model_better.h5')
print('model loaded')
##print(model.summary)

def identifyGesture(img, count):
    cv2.imwrite('images/{}.jpeg'.format(count), img)
    size_img = 200,200
    img = cv2.resize(img, size_img)
    cv2.imwrite('images/{}_1.jpeg'.format(count), img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.reshape((1,200,200,3))
    prediction = model.predict_classes(img)
    key = (key for key, value in classes.items() if value == prediction[0])
    return(list(key)[0])



# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Tone Selector')
##cv2.namedWindow('Hand')
##cv2.namedWindow('HandTrain')

cap = cv2.VideoCapture(0)

# previous values of cropped variable
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

# previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
prevcnt = np.array([], dtype=np.int32)

# gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
# gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
# 10 frames till gestureDetected decrements to 0.
gestureStatic = 0
gestureDetected = 0

def nothing(x):
    pass

### TrackBars for fixing skin color of the person
cv2.createTrackbar('H for min', 'Tone Selector', 0, 255, nothing)
cv2.createTrackbar('S for min', 'Tone Selector', 0, 255, nothing)
cv2.createTrackbar('V for min', 'Tone Selector', 0, 255, nothing)
cv2.createTrackbar('H for max', 'Tone Selector', 0, 255, nothing)
cv2.createTrackbar('S for max', 'Tone Selector', 0, 255, nothing)
cv2.createTrackbar('V for max', 'Tone Selector', 0, 255, nothing)

cv2.setTrackbarPos('H for min', 'Tone Selector', 0)
cv2.setTrackbarPos('S for min', 'Tone Selector', 48)
cv2.setTrackbarPos('V for min', 'Tone Selector', 80)
cv2.setTrackbarPos('H for max', 'Tone Selector', 20)
cv2.setTrackbarPos('S for max', 'Tone Selector', 255)
cv2.setTrackbarPos('V for max', 'Tone Selector', 255)

count = 0

while(True):

    # Getting min and max colors for skin

    lower = np.array([cv2.getTrackbarPos('H for min', 'Tone Selector'),
                      cv2.getTrackbarPos('S for min', 'Tone Selector'),
                      cv2.getTrackbarPos('V for min', 'Tone Selector')], np.uint8)
    upper = np.array([cv2.getTrackbarPos('H for max', 'Tone Selector'),
                      cv2.getTrackbarPos('S for max', 'Tone Selector'),
                      cv2.getTrackbarPos('V for max', 'Tone Selector')], np.uint8)
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (400,400))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##    converted = cv2.GaussianBlur(converted, (5, 5), 0)
    skinRegion = cv2.inRange(converted, lower, upper)

    # Do contour detection on skin region
    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if(len(contours)==0):
        print('bad')
    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]

    stencil = np.zeros(frame.shape).astype(frame.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cnt], color)
    handTrainImage = cv2.bitwise_and(frame, stencil)

    if (ret > 0.70):
        gestureStatic = 0
    else:
        gestureStatic += 1

    # crop coordinates for hand.
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    cv2.rectangle(frame, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # if the crop area has changed drastically form previous frame, update it.
    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

##        print('Hand Gesture Changed!')

    handImage = frame.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    # Training image with black background
    handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                     max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

    if gestureStatic == 10:
        gestureDetected = 10;
        print("Gesture Detected")
        letterDetected = identifyGesture(handTrainImage, count)
        print(letterDetected)
        count+=1
        
    if gestureDetected > 0:
        if (letterDetected != None):
            cv2.putText(frame, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        gestureDetected -= 1
        
    cv2.imshow('Camera Output', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
