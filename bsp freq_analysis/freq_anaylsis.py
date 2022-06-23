# loading and playing video

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack.basic import ifft
from ref_calculate import y_ref,x_ref
from scipy import pi
from scipy.fftpack import fft, fftfreq



fileName='rod1_4sec.mp4'  # change the file name if needed

cap = cv2.VideoCapture(fileName)   


i=0
j=0
# for calculating the reference along y axis.
y_reff=y_ref(fileName)
# for calculating the reference along x-axis.
x_reff=x_ref(fileName)



# to store frame no.
x=[]

# to store the y-deflection from mean position.
y_deflect=[]

# to store the x-deflection from mean position.
x_deflect=[]


 # play the video by reading frame by frame
while(cap.isOpened()):                   
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
        
        dis_y=0
        dis_x=0
        
        a_y=0
        a_x=0
        
        for line in lines:
            for x1,y1,x2,y2 in line: 
                #if j==0:
                    #x_ref=(y2+y1)/2
                    #j=j+1
                    #print(x1,"  ",y1)

                if k==1:
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
                    dis_y=((y2+y1)/2)-y_reff
                    dis_x=((x2+x1)/2)-x_reff
                    a_y=(y2+y1)/2
                    a_x=((x2+x1)/2)
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
        pr=str(dis_y)
        new_image = cv2.putText(img = lines_edges ,
            text = "pixel's deviation:- "+pr,
            org = (200, 200),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1.0,
            color = (255, 0, 0),
            thickness = 2
            )

        
      #  cv2.imwrite('rod_new'+str(i)+'.jpg',new_image)
        cv2.imshow('rod_new',new_image)
        i+=1

        
        x.append(i)
        y_deflect.append(dis_y)
        x_deflect.append(dis_x)
        
        
        print(i,"-> ",a_y," -- ",y_reff,"=",dis_y,"   the x delfection",a_x," -- ",x_reff,"=",dis_x )
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


"""
# Plotting both the curves simultaneously
plt.plot(x, y_deflect, color='r', label='y-axis vibration')
plt.plot(x, x_deflect, color='g', label='x-axis vibration')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("frames")
plt.ylabel("deflection from mean postion")
plt.title("rod vibration")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

# to plot subplots
figure, axis = plt.subplots(2,2)
  
# For Sine Function
axis[0, 0].plot(x,y_deflect)
axis[0, 0].set_title("rod vibration in y-axis")
  
# For Cosine Function
axis[0, 1].plot(x,x_deflect)
axis[0, 1].set_title("rod vibration in x-axis")

# Combine all the operations and display
plt.show()
"""
#simple ploting the graph
plt.plot(x,y_deflect)
plt.title('rod vibration _y')
plt.xlabel('frames')
plt.ylabel('deflection from reference')
plt.show()

plt.plot(x,x_deflect)
plt.title('rod vibration _x')
plt.xlabel('frames')
plt.ylabel('deflection from reference')
plt.show()




sample_rate = 30
#sample_rate is defined as number of samples taken per second
N = (4 - 0) * sample_rate

frequency = np.linspace (0.0, 15, int (N/2))


freq_data = fft(y_deflect)
#freq_data=np.abs(freq_data)
y = 2/N * np.abs (freq_data [0:np.int (N/2)])

plt.plot(frequency,y)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


"""
sample_rate = 30
#sample_rate is defined as number of samples taken per second
N = (4 - 0) * sample_rate

frequency = np.linspace (0.0, 15, int (N/2))


freq_data = fft(y_deflect)
#freq_data=np.abs(freq_data)
y = 2/N * np.abs (freq_data [0:np.int (N/2)])

plt.plot(x, freq_data)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


i=0
for k in freq_data:
    if(k<170):
        freq_data[i]=0
    i=i+1

inverse_freq=ifft(freq_data)

plt.plot(x, inverse_freq)
plt.title('reconstruct domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()
"""


cap.release()



cv2.destroyAllWindows()