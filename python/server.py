from socket import socket, AF_INET, SOCK_DGRAM
import cv2


HOST = ''
PORT = 12345

s = socket(AF_INET, SOCK_DGRAM)
s.bind((HOST, PORT))

if __name__ == '__main__':

    while True:

        msg, addr = s.recvfrom(8192)
        print(msg, addr)
    
        if cv2.waitKey(1) == ord('q'):
            break