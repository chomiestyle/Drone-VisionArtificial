from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse
import imutils
import math

# standard argparse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-sz', '--SafeZone', type=int, default=1,
    help='use -sz to change the safeZone value of the drone. Range 0-3')
#parser.add_argument('-sx', '--saftey_x', type=int, default=100,
#    help='use -sx to change the saftey bound on the x axis . Range 0-480')
#parser.add_argument('-sy', '--saftey_y', type=int, default=55,
#    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-os', '--override_speed', type=int, default=1,
    help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

# Speed of the drone
S = 20
S2 = 5
UDOffset = 150

# this is just the bound box sizes that openCV spits out *shrug*
#faceSizes = [1026, 684, 456, 304, 202, 136, 90]
body_distance=[10,20,30,40,50,60]

#Different Safe zones
Safe_Zones=[(10,30,2),(90,200,15),(120,300,60),(140,330,100)]
speeds=[2,5,10,15,20]

# These are the values in which kicks in speed up mode, as of now, this hasn't been finalized or fine tuned so be careful
# Tested are 3, 4, 5
acc = [500,250,250,150,110,70,50]

# Frames per second of the pygame window display
FPS = 25
dimensions = (960, 720)
frames_to_ignore = 30*5
ignored_frames = 0
# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# If we are to save our sessions, we need to make sure the proper directories exist
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

class FrontEnd(object):
    
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.safe_zone = args.SafeZone
        self.oSpeed = args.override_speed
        self.send_rc_control = False

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        should_stop = False
        imgCount = 0
        OVERRIDE = False
        self.tello.get_battery()

        result = cv2.VideoWriter('Prueba_Tello.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, dimensions)
        vid = self.tello.get_video_capture()



        if args.debug:
            print("DEBUG MODE ENABLED!")
        while not should_stop:
            #self.update()
            #time.sleep(1 / FPS)

            # Listen for key presses
            k = cv2.waitKey(20)
            self.Set_Action(k,OVERRIDE)


            if OVERRIDE:
                self.Action_OVERRRIDE(k)

            # Quit the software
            if k == 27:
                should_stop = True
                break
            if vid.isOpened():

                ret, image = vid.read()
                
                if ignored_frames < frames_to_ignore:
                    ignored_frames += 1
                    continue
                 
                imgCount+=1
                if imgCount % 30 != 0:
                    continue
                if ret:
                    image = imutils.resize(image,width=min(350, image.shape[1]))

                    # Detecting all the regions
                    # in the Image that has a
                    # pedestrians inside it

                    (regions, weigths) = hog.detectMultiScale(image,
                                                winStride=(4, 4),
                                                padding=(4, 4),
                                                scale=1.1)



                    # Safety Zone Z
                    szZ = Safe_Zones[self.safe_zone][2]
                    # Safety Zone X
                    szX = Safe_Zones[self.safe_zone][0]
                    # Safety Zone Y
                    szY = Safe_Zones[self.safe_zone][1]

                    center = (int(image.shape[1] / 2), int(image.shape[0] / 2))

                    # if we've given rc controls & get body coords returned
                    if self.send_rc_control and not OVERRIDE:
                        if len(regions) > 0:
                            max_index = np.where(weigths == np.amax(weigths))
                            (x, y, w, h) = regions[max_index[0][0]]
                            cv2.rectangle(image, (x, y),
                                        (x + w, y + h),
                                        (0, 0, 255), 2)
                            # points

                            person = (int(x + w / 2), int(y + h / 2))
                            distance = (person[0] - center[0], center[1] - person[1])
                            theta0 = 86.6
                            B = image.shape[1]
                            person_width = w

                            # z
                            theta1 = theta0 * (2 * abs(distance[0]) + person_width) / (2 * B)
                            z = (2 * abs(distance[0]) + person_width) / (2 * math.tan(math.radians(abs(theta1))))
                            z = int(z) - 229
                            distance = (int(distance[0]), int(distance[1]), int(z))

                            if not args.debug:
                                # for turning
                                
                                if distance[0] < -szX:
                                    self.yaw_velocity = S
                                    # self.left_right_velocity = S2
                                elif distance[0] > szX:
                                    self.yaw_velocity = -S
                                    # self.left_right_velocity = -S2
                                else:
                                    self.yaw_velocity = 0
                        
                                # for up & down
                                if distance[1] > szY:
                                    self.up_down_velocity = S
                                elif distance[1] < -szY:
                                    self.up_down_velocity = -S
                                else:
                                    self.up_down_velocity = 0

                                F = 0
                                if abs(distance[2]) > acc[self.safe_zone]:
                                    F = S

                                # for forward back
                                if distance[2] > szZ:
                                    self.for_back_velocity = S + F
                                elif distance[2] < -szZ:
                                    self.for_back_velocity = -S - F
                                else:
                                    self.for_back_velocity = 0

                            cv2.line(image, center, (center[0] + distance[0], center[1]), (255, 0, 0))
                            cv2.line(image, (center[0] + distance[0], center[1]), person, (0, 255, 0))
                            cv2.circle(image, center, 3, (0, 255, 0))
                            cv2.circle(image, person, 3, (0, 0, 255))

                            cv2.putText(image, "d:" + str(distance), (0, 20), 2, 0.7, (0, 0, 0), )
                        # if there are no body detected, don't do anything
                        else:
                            self.yaw_velocity = 0
                            self.up_down_velocity = 0
                            self.for_back_velocity = 0
                            print("NO TARGET")
                    
                        self.update()

                    dCol = lerp(np.array((0,0,255)),np.array((255,255,255)),self.safe_zone+1/7)

                    if OVERRIDE:
                        show = "OVERRIDE: {}".format(self.oSpeed)
                        dCol = (255,255,255)
                    else:
                        show = "AI: {}".format(str(self.safe_zone))
                    cv2.rectangle(image, (int(center[0] - szX / 2), int(center[1] - szY / 2)),
                                  (int(center[0] + szX / 2), int(center[1] + szY / 2)), (255, 0, 0), 1)
                    # Showing the output Image
                    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
                    # Write the frame into the
                    # file 'Prueba_Tello.avi'
                    result.write(image)

                    # Draw the distance choosen
                    cv2.putText(image, show, (32, 664), cv2.FONT_HERSHEY_SIMPLEX, 1, dCol, 2)

                    # Display the resulting frame
                    cv2.imshow(f'Tello Tracking...', image)
                else:
                    break

        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()
        result.release()
        # Call it always before finishing. I deallocate resources.
        self.tello.end()


    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)
    def Action_OVERRRIDE(self,k):
        # S & W to fly forward & back
        if k == ord('w'):
            self.for_back_velocity = int(S * oSpeed)
        elif k == ord('s'):
            self.for_back_velocity = -int(S * oSpeed)
        else:
            self.for_back_velocity = 0

        # a & d to pan left & right
        if k == ord('d'):
            self.yaw_velocity = int(S * self.oSpeed)
        elif k == ord('a'):
            self.yaw_velocity = -int(S * self.oSpeed)
        else:
            self.yaw_velocity = 0

        # Q & E to fly up & down
        if k == ord('e'):
            self.up_down_velocity = int(S * self.oSpeed)
        elif k == ord('q'):
            self.up_down_velocity = -int(S * self.oSpeed)
        else:
            self.up_down_velocity = 0

        # c & z to fly left & right
        if k == ord('c'):
            self.left_right_velocity = int(S * self.oSpeed)
        elif k == ord('z'):
            self.left_right_velocity = -int(S * self.oSpeed)
        else:
            self.left_right_velocity = 0
        return
    def Set_Action(self,k, OVERRIDE):
        # Press 0 to set distance to 0
        if k == ord('0'):
            if not OVERRIDE:
                print("Distance = 0")
                self.safe_zone = 0

        # Press 1 to set distance to 1
        if k == ord('1'):
            if OVERRIDE:
                self.oSpeed = 1
            else:
                print("Distance = 1")
                self.safe_zone = 1

        # Press 2 to set distance to 2
        if k == ord('2'):
            if OVERRIDE:
                self.oSpeed = 2
            else:
                print("Distance = 2")
                self.safe_zone = 2

        # Press 3 to set distance to 3
        if k == ord('3'):
            if OVERRIDE:
                self.oSpeed = 3
            else:
                print("Distance = 3")
                self.safe_zone = 3

        # Press 4 to set distance to 4
        if k == ord('4'):
            if not OVERRIDE:
                print("Distance = 4")
                self.safe_zone = 4

        # Press 5 to set distance to 5
        if k == ord('5'):
            if not OVERRIDE:
                print("Distance = 5")
                self.safe_zone = 5

        # Press 6 to set distance to 6
        if k == ord('6'):
            if not OVERRIDE:
                print("Distance = 6")
                self.safe_zone= 6

        # Press T to take off
        if k == ord('t'):
            if not args.debug:
                print("Taking Off")
                self.tello.takeoff()
                self.tello.get_battery()
            self.send_rc_control = True

        # Press L to land
        if k == ord('l'):
            if not args.debug:
                print("Landing")
                self.tello.land()
            self.send_rc_control = False

        # Press Backspace for controls override
        if k == 8:
            if not OVERRIDE:
                OVERRIDE = True
                print("OVERRIDE ENABLED")
            else:
                OVERRIDE = False
                print("OVERRIDE DISABLED")


def lerp(a,b,c):
    return a + c*(b-a)

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
