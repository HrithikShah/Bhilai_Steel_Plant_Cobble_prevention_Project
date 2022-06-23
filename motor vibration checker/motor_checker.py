# loading and playing video

import numpy as np
import cv2
import tensorflow as tf
import time

fileName='motor_vibe_Trim.mp4'  # change the file name if needed

cap = cv2.VideoCapture(fileName)   




i=0
while(cap.isOpened()):                    # play the video by reading frame by frame
    # READING OF FRAME
    ret, frame1 = cap.read()  # first image
    time.sleep(1/25)          # slight delay
    ret, frame2 = cap.read()  # second image 
    
    if ret==True:
        # optional: do some image processing here 

        # CROP FRAME 1

        # Window name in which image is displayed
        window_name = 'Image'
        
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (650, 450)
        
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (1000, 800)
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        #thickness = 10
        croped_frame1=frame1[450:800 , 650:1000]
        croped_frame2=frame2[450:800 , 650:1000]

        # rectangle drawing
        start_point = (650,450)
        end_point = (1000, 800)
        color = (0,0,255)
        thickness = 5  

        image_final = cv2.rectangle(frame2, start_point, end_point, color, thickness)

        # ssim comparision
        ssim_score= tf.image.ssim(  frame1, frame2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

        # print("the SSIM score =", ssim_score)
        print(float(ssim_score))
        ssim=str(float(ssim_score))
        # Using cv2.putText()
    
        new_image = cv2.putText(img = image_final ,
            text = "ssim score:- "+ssim,
            org = (200, 200),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 3.0,
            color = (255, 0, 0),
            thickness = 3
            )
     
        cv2.imshow('frame',new_image)  
         #saving frame
        #cv2.imwrite('motor'+str(i)+'.jpg',new_image)
        i+=1

        

                    # show the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()