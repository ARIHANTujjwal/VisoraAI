#https://www.youtube.com/watch?v=eDIj5LuIL4A
"""
Lesson1

    Images are numpy arrays

    x = cv2.imread(image) - loads image

    .shape - gives image hight, width, number of channels
    x.shape
        - channels represent pixel color values

    pixel values from 0 - 255
    binary image pixel value 0 - 1 (0 or 255)
----------------------------
Lesson2

    reading image:
        - image_path = os.path.join('.', 'data', 'bird.jpg')
            - '.' represents the current directory where the program is being run
            - 'data' represents the sub folder in the directory where the file is house
            - 'bird.jpg' is the exact file which needs to be read
        - img = cv2.imread(image_path)
    write image:
        - cv2.imwrite(os.path.join('.','data', 'bird_out.jpg'), img)
            - used to save an image to a specified file
    visulize image:
        - cv2.imshow('image', img)
        - cv2.waitKey(0), durration in brackets is in milliseconds
------
    Video:
        - video_path = os.path.join('.','data','monkey.mp4')
        - video = cv2.VideoCapture(video_path)
        - ret, frame = video.read()
                - ret is a boolean value of if the frame was captured successfully
                - frame is the actual path of the frame
    

    video.release() - release memory over the thing
    cv2.destroyAllWindows()
-----
    Webcam:
        - webcam = cv2.VideoCapture(0)
        - 0xFF == ord('q') when user clicks q 
------------------
Lesson3:
    Resizing:
        - img = cv2.imread(os.path.join('.','dog.jpg'))
        - img.shape --> returns the pixel length of the piage in terms of x,y,# of brightness channels
        - resized_img = cv2.resize(img,(640, 480)) ---> if this were printed, x = 480 while y = 640
    Cropping:
        - img = cv2.imread(os.path.join('.','dogs.jpg'))
        - 


"""
import cv2

import os

every = cv2.VideoCapture(0)

