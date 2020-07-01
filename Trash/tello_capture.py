import time
import cv2

def Video_capture_Tello(sock,tello_address,mypc_address):

    print("Start connection streaming")
    capture = cv2.VideoCapture('udp:/0.0.0.0:11111', cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.open('udp:/0.0.0.0:11111')
    return capture


