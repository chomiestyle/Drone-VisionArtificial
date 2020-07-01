import cv2 
import imutils 
import math
import numpy as np
# Initializing the HOG person 
# detector 


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('pruebaeduardo.mp4')

frame_count = 0
while cap.isOpened(): 
    frame_count +=1
    
    
    # Reading the video stream 
    ret, image = cap.read() 
    
    if frame_count%2 != 0:
        continue


    if ret: 
        image = imutils.resize(image,  
                               width=min(350, image.shape[1]))
 
        #image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE) 
    
        # Detecting all the regions  
        # in the Image that has a  
        # pedestrians inside it 
        
        (regions, weigths) = hog.detectMultiScale(image,
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.1) 


        difference = None
        if len(regions)>0:
            max_index =np.where(weigths == np.amax(weigths))
            (x, y, w, h) =  regions[max_index[0][0]]
            cv2.rectangle(image, (x, y), 
                            (x + w, y + h),  
                            (0, 0, 255), 2) 
            
            #points
            center = (int(image.shape[1]/2) , int(image.shape[0]/2))
            person = (int(x + w/2), int(y + h/2))
            distance = (person[0] - center[0],   center[1] - person[1])
            theta0 = 86.6
            B = image.shape[1]
            person_width = w

            #z
            theta1 = theta0 * ( 2 * abs(distance[0]) + person_width) / (2*B)

            z =  ( 2 * abs(distance[0]) + person_width) /(2 * math.tan(math.radians(abs(theta1))))
            z_difference = 229 - int(z)
            difference = (int(distance[0]), int(distance[1]), z_difference)
            
            x_difference = difference[0]
            y_difference = difference[1]
            z_difference = difference[2]

            safe_zone_radius = (10,10,5)
            velocity = 15

            #velocities
            #theta2 = theta0 * ( 2 * abs(difference[0])) / (2*B)
            
            in_safe_zone_x = abs(x_difference) < safe_zone_radius[0]
            x_sign = x_difference/abs(x_difference) if x_difference != 0 else 0
            x_velocity = 0 if in_safe_zone_x else velocity * x_sign

            in_safe_zone_y= abs(y_difference) < safe_zone_radius[1]
            y_sign = y_difference / abs(y_difference) if y_difference != 0 else 0
            y_velocity = 0 if in_safe_zone_y else velocity * y_sign
        
            in_safe_zone_z= abs(z_difference) < safe_zone_radius[2]
            z_sign = z_difference / abs(z_difference) if z_difference != 0 else 0
            z_velocity = 0 if in_safe_zone_z else velocity * z_sign

            #send_rc_control(self, x_velocity, z_velocity , y_velocity, 0):

            print (in_safe_zone_x, in_safe_zone_y, in_safe_zone_z)

            
            ###dibujos
            cv2.line(image, center, (center[0]+difference[0], center[1]), (255, 0, 0))
            cv2.line(image, (center[0] + difference[0], center[1]), person, (0, 255, 0))
            cv2.circle(image, center,3, (0, 255, 0))
            cv2.circle(image, person,3, (0, 0, 255))
            

        ####
        # Showing the output Image 
        image = imutils.resize(image,  
                               width=750)
        cv2.putText(image,"d:" + str(difference), (0,20), 2,0.7, (0,255,0),)
        
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break
  
cap.release() 
cv2.destroyAllWindows() 