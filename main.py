"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
from random import randint # for test

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import numpy as np

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client =  mqtt.Client(client_id="", clean_session=True, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL, bind_address="")
    return client

##Start of my own func
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bbox(frame, results, args, width, height):
    class_lbls = []
    #Draw bound box onto the objects
    for obj in results[0][0]:
        conf = obj[2]
        if conf >= 0.5:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            class_id = int(obj[1])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.prob_threshold , 1)
            class_label = CLASSES[class_id]
            class_lbls.append(class_label)

    #Prepare for FFmpeg
    classes = cv2.resize(results[0].transpose((1,2,0)), (width,height), interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes, class_lbls #frame, unique_classes #class_label #, class_id

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes

##End of my own func

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    mqtt_client = client
    #mqtt_client.loop_start()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    log.info("Loading the model through Inference Engine...")
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #set arr for tracking objects
    ts = [0,0,False] #[time_first, time_last, got_first]
    total_count = 0
    person_found = []

    #out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        retval, frame = cap.read()
        if not retval:
            break
        key_pressed = cv2.waitKey(60) #wait for 60 ms

        ### TODO: Pre-process the image as needed ###
        pr_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        pr_frame = pr_frame.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_frame = pr_frame.reshape(1, *pr_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(pr_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:


            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            #output_frame, classes, class_lbl = draw_bbox(frame, results, args, width, height)

            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            #out.write(frame)
            #out_frame, classes = draw_masks(result, width, height)
           
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            '''
            for idx in class_lbl:
                if idx == "person":
                    total_count += total_count
                    if ts[2] == False:
                        ts[0] = time.time()
                        ts[2] = True
                else:
                    if ts[2] == True:
                        ts[1] = time.time()
                        ts[2] = False
                person_found.append([ts[0], ts[1]])

           # for list in person_found:
            #    duration = list[1] - list[0]
            '''          
            duration = time.time()
            count = int(5)
            total_count = int(6)

            mqtt_client.publish("person", json.dumps({"total": total_count}))
            mqtt_client.publish("person", json.dumps({"count": count}))
            mqtt_client.publish("person/duration", json.dumps({"duration": duration}))

        ### TODO: Send the frame to the FFMPEG server ###
            
            #ff_frame = np.dstack((output_frame, output_frame, output_frame))
            #ff_frame = np.uint8(ff_frame)
            #ff_frame = classes * (255/20)
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
            
        ### TODO: Write an output image if `single_image_mode` ###
    
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
