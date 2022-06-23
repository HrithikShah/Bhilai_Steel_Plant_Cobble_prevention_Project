# loading and playing video

import numpy as np
import cv2


def y_ref(file):

    #fileName='rood2.mp4'  # change the file name if needed

    cap = cv2.VideoCapture(file)   


    i=0
    y_ref=0

    while(cap.isOpened()):                    # play the video by reading frame by frame
        # READING OF FRAME
        ret, frame1 = cap.read()  # first image
        
        
        if ret==True:
            # optional: do some image processing here 
            #croped_frame1=frame1[120:230 , 90:400]
            #convertig to req frame
            croped_frame1=frame1[120:230 , 90:400]
            #print(croped_frame1.shape)
            gray = cv2.cvtColor(croped_frame1, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image to binary
            ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

            # line formation
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(binary,(kernel_size, kernel_size),0)
                
            #canny
            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

            #making line with hougep
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 15  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 50  # minimum number of pixels making up a line
            max_line_gap = 20  # maximum gap in pixels between connectable line segments
            line_image = np.copy(croped_frame1) * 0  # creating a blank to draw lines on


            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
            
            
            k=1
            dis=0
            
            for line in lines:
                for x1,y1,x2,y2 in line: 

                    if k==1:
                        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
                        dis=((y2+y1)/2)
                        y_ref=y_ref+dis
                    #if i==2:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)
                    #if i==3:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
                    #if i==4:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                    k+=1
                    #dis=((x1+x2)/2)-x_ref


            lines_edges = cv2.addWeighted(croped_frame1, 0.8, line_image, 1, 0)
            # Using cv2.putText()
            pr=str(dis)
            new_image = cv2.putText(img = lines_edges ,
                text = "pixel's deviation:- "+pr,
                org = (200, 200),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1.0,
                color = (255, 0, 0),
                thickness = 2
                )
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()



    cv2.destroyAllWindows()
   

    return y_ref/i


def x_ref(file):

    #fileName='rood2.mp4'  # change the file name if needed

    cap = cv2.VideoCapture(file)   


    i=0
    x_ref=0

    while(cap.isOpened()):                    # play the video by reading frame by frame
        # READING OF FRAME
        ret, frame1 = cap.read()  # first image
        
        
        if ret==True:
            # optional: do some image processing here 
            #croped_frame1=frame1[120:230 , 90:400]
            #convertig to req frame
            croped_frame1=frame1[120:230 , 90:400]
            #print(croped_frame1.shape)
            gray = cv2.cvtColor(croped_frame1, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image to binary
            ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

            # line formation
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(binary,(kernel_size, kernel_size),0)
                
            #canny
            low_threshold = 50
            high_threshold = 150
            edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

            #making line with hougep
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 15  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 50  # minimum number of pixels making up a line
            max_line_gap = 20  # maximum gap in pixels between connectable line segments
            line_image = np.copy(croped_frame1) * 0  # creating a blank to draw lines on


            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
            
            
            k=1
            dis=0
            
            for line in lines:
                for x1,y1,x2,y2 in line: 

                    if k==1:
                        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
                        dis=((x2+x1)/2)
                        x_ref=x_ref+dis
                    #if i==2:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)
                    #if i==3:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
                    #if i==4:
                        #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                    k+=1
                    #dis=((x1+x2)/2)-x_ref


            lines_edges = cv2.addWeighted(croped_frame1, 0.8, line_image, 1, 0)
            # Using cv2.putText()
            pr=str(dis)
            new_image = cv2.putText(img = lines_edges ,
                text = "pixel's deviation:- "+pr,
                org = (200, 200),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1.0,
                color = (255, 0, 0),
                thickness = 2
                )

            

            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()



    cv2.destroyAllWindows()
   

    return x_ref/i

    