import socket
import cv2


def send_data(ip='127.0.0.1', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # message = f'{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}'
    message = 'sending...'
    sock.sendto(message.encode(), (ip, port))
    sock.close()


if __name__ == '__main__':

    while True:
        send_data(ip='192.168.0.29')
        # send_data(ip='192.168.0.37')
        # print('sending...')
    
        if cv2.waitKey(1) == ord('q'):
            break