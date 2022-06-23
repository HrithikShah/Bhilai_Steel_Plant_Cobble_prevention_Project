# loading and playing video

import numpy as np
import cv2

from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

fileName='spark_Trim.mp4'  # change the file name if needed
"""
model = core.Model.load("model_weights.pth", ["spark"])

frame1 = utils.read_image("spark51.jpg") 
predictions = model.predict(frame1)
labels, boxes, scores = predictions
        #show_labeled_image(frame1, boxes, labels)
thresh=0.8
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]

#format of box is  [ x1, y1,  x2  ,y2]

print(filtered_boxes)


image=frame1
for box in filtered_boxes:
        x1,y1,x2,y2=box
        start_point = (int(x1), int(y1))
  
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (int(x2), int(y2))
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        image = cv2.rectangle(frame1, start_point, end_point, color, thickness)
  
cv2.imshow("window_name", image)
cv2.imwrite('spark.jpg',image)
#show_labeled_image(frame1, filtered_boxes, filtered_labels)
       
"""

model = core.Model.load("model_weights.pth", ["spark"])
cap = cv2.VideoCapture(fileName)   
i=0
while(cap.isOpened()):                    # play the video by reading frame by frame
    # READING OF FRAME
    ret, frame1 = cap.read()  # first image
    
    image=frame1
    if ret==True:
      
        predictions = model.predict(frame1)
        labels, boxes, scores = predictions
        #show_labeled_image(frame1, boxes, labels)

        thresh=0.7
        filtered_indices=np.where(scores>thresh)
        #filtered_scores=scores[filtered_indices]
        filtered_boxes=boxes[filtered_indices]
        #num_list = filtered_indices[0].tolist()
        #filtered_labels = [labels[i] for i in num_list]
        #show_labeled_image(frame1, filtered_boxes, filtered_labels)


        
        image=frame1
        for box in filtered_boxes:

                x1,y1,x2,y2=box
                start_point = (int(x1), int(y1))
        
                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                end_point = (int(x2), int(y2))
                
                # Blue color in BGR
                color = (0, 255 , 0)
                
                # Line thickness of 2 px
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        cv2.imshow("window_name", image)

        #cv2.imwrite('spark'+str(i)+'.jpg', image)

        i=i+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()