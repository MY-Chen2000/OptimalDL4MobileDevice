
import socket
import threading
import sys
import os
import datetime,time
import struct
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

classes=read_class_names('./coco.names')


input_size   = 416
image_path   = "./docs/kite.jpg"

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
model.summary()

now = datetime.datetime.now()

pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')





image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()

now2 = datetime.datetime.now()

print((now2 - now).seconds)

def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.18.15.27', 8000))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print('Waiting connection...')

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))
    # conn.settimeout(500)
    conn.send('Hi, Welcome to the server!'.encode("utf-8"))

    while 1:

        fileinfo_size = struct.calcsize('128sq')
        buf = conn.recv(fileinfo_size)
        if buf:
            # filename, filesize = struct.unpack('128sq', buf)
            filesize=buf
            fn = 'rgb.jpg'#filename.strip(b"\x00").decode("utf-8")
            new_filename = os.path.join('./', 'new_' + fn)
            print(new_filename, filesize)
            print('file new name is {0}, filesize if {1}'.format(new_filename, filesize))

            recvd_size = 0  # 定义已接收文件的大小
            fp = open(new_filename, 'wb')
            print('start receiving...')

            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print('end receive...')
            tic = time.time()
            original_image = cv2.imread(new_filename)
            # rotate the image
            height, width = original_image.shape[:2]

            matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), -90, 1)
            dst = cv2.warpAffine(original_image, matRotate, (width, height * 2))
            rows, cols = dst.shape[:2]

            for col in range(0, cols):
                if dst[:, col].any():
                    left = col
                    break

            for col in range(cols - 1, 0, -1):
                if dst[:, col].any():
                    right = col
                    break

            for row in range(0, rows):
                if dst[row, :].any():
                    up = row
                    break

            for row in range(rows - 1, 0, -1):
                if dst[row, :].any():
                    down = row
                    break

            res_widths = abs(right - left)
            res_heights = abs(down - up)
            res = np.zeros([res_heights, res_widths, 3], np.uint8)

            for res_width in range(res_widths):
                for res_height in range(res_heights):
                    res[res_height, res_width] = dst[up + res_height, left + res_width]
            original_image = res
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]

            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            pred_bbox = model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')

            strbox = str(len(bboxes)) + ','
            for i in range(0, len(bboxes)):
                for j in range(0, 5):
                    strbox = strbox + str(bboxes[i][j]) + ','
                strbox = strbox + classes[int(bboxes[i][5])] + ','

            image = utils.draw_bbox(original_image, bboxes)
            image = Image.fromarray(image)
            image.show()

            toc = time.time()
            print(toc - tic)
        conn.send(strbox.encode("utf-8"))
        print(conn.recv(1024).decode('utf-8'))
        conn.close()
        break



if __name__ == '__main__':
    socket_service()


