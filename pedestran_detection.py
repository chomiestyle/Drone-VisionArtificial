import cv2 
import imutils 
import math
from Trash import tello_capture
from djitellopy import Tello



#Create the socket
tello = Tello()
tello.connect()
tello.streamoff()

"""
tello_ip = '192.168.10.1'
host = '192.168.10.3'
port = 8890
tello_port = 8889
tello_address = (tello_ip, tello_port)
mypc_address= (host, port)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

"""
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
mypc_address= (tello.local_ip, tello.local_port)
cap= tello_capture.Video_capture_Tello(tello.socket, tello.tello_address, mypc_address)
tello.takeoff()

#cap = cv2.VideoCapture('prueba_2.mp4')

frame_count = 0
while cap.isOpened(): 
    frame_count +=1
    
    
    # Reading the video stream 
    ret, image = cap.read() 
    
    if frame_count%2 != 0:
        continue

    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    if ret: 
        image = imutils.resize(image,  
                               width=min(350, image.shape[1]))

        #image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE) 
    
        # Detecting all the regions  
        # in the Image that has a  
        # pedestrians inside it 
        
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.1) 
    
    
    
        # Drawing the regions in the  
        # Image 
        for (x, y, w, h) in regions: 
            cv2.rectangle(image, (x, y), 
                          (x + w, y + h),  
                          (100, 100, 100), 1) 

        distance = None
        if len(regions)>0:
       
            (x, y, w, h) =  regions[0]
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
            distance = (int(distance[0]), int(distance[1]), int (z))

            
            ###dibujos
            cv2.line(image, center, (center[0]+distance[0], center[1]), (255, 0, 0))
            cv2.line(image, (center[0] + distance[0], center[1]), person, (0, 255, 0))
            cv2.circle(image, center,3, (0, 255, 0))
            cv2.circle(image, person,3, (0, 0, 255))
            

        ####
        # Showing the output Image 
        image = imutils.resize(image,  
                               width=350)
        cv2.putText(image,"d:" + str(distance), (0,20), 2,0.7, (0,0,0),)
        
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break
  
cap.release() 
cv2.destroyAllWindows() 