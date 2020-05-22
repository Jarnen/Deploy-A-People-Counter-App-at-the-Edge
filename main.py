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
import time
from datetime import datetime

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
# detect
# Reference: Most of this idea are taken from Foundation
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# count number of persons in current frame and draw boxes around them
def count_draw(frame, result, args, width, height):
    p_counts = 0 # set p_counts for every frame and recounts
    '''
    Draw bounding boxes onto each frame.
    '''
    for obj in result[0][0]: # Output shape is 1x1x100x7/1x1xNx7
        conf = obj[2]
        if conf >= args.prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            class_id = int(obj[1])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            class_label = CLASSES[class_id]
            if class_label == "person":
                p_counts += 1
    return frame, p_counts

def get_uclasses(result, width, height):
    '''
    Get unique classes from the result.
    '''
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)

    return unique_classes

total_count = 0 # to add total persons
had_found = False # to check if peron is in the frame
had_counted = False # to check if person was counted
appearanceFrom = datetime.now() # reference time in order to wait 3s

# check and count total of persons entering and exiting one at a time
def get_total():
    global total_count
    global appearanceFrom
    global had_found 
    global had_counted

    checkNow = datetime.now()
    timeDif = checkNow - appearanceFrom
    if had_found == False:
        had_found = True
        had_counted = False 
        appearanceFrom = checkNow
    elif timeDif.seconds >= 3 and had_counted == False: #add person in frame more than 3s'
        total_count = total_count + 1
        had_counted = True
    return total_count, timeDif.seconds
    
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

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold


    ### TODO: Load the model through `infer_network` ###
    log.info("Loading the model through Inference Engine...")
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Set flag for the input image
    single_image_mode = False

    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    #cap = cv2.VideoCapture(args.input)
    #cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))

    ## Define and set global variables
    global had_found
    global total_count
    duration = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
       
        ### TODO: Read from the video capture ###
        # get return value and frame
        retval, frame = cap.read()

        if not retval:
            break
        key_pressed = cv2.waitKey(60) #wait for 60 ms

        ### TODO: Pre-process the image as needed ###
        pr_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        pr_frame = pr_frame.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_frame = pr_frame.reshape(1, *pr_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(pr_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            # get and draw the bounding box for person
            frame, p_counts = count_draw(frame, result, args, width, height)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # get unique class from the frame 
            # because our scenario is one person entering frame and exiting at a time.
            unique_classes = get_uclasses(result, width, height)
            
            # check if the ID (15) of persons enters frame (present)
            # and use function get_total to calculate and get total count and duration
            if 15 in unique_classes: 
                total_count, duration = get_total()
            
            # check if the ID (15) of persons exits the frame (absent)
            # set alreadyFound to False if person already counted and already found
            # and publish the person/duration to MQTT Server with duration and total count
            elif 15 not in unique_classes and had_counted == True and had_found == True:
                client.publish("person/duration", json.dumps({"duration": duration}))
                had_found = False
            
            # otherwise, set alreadyFound to false
            else:
                if had_found == True:
                    log.info("Person counted already...")
                    had_found = False
            
            # Draw performance stats on the frame
            total_message = "The Total Count: {}".format(total_count)
            current_message = "The Current Count: {}".format(p_counts)
            duration_message = "Duration in Frame: {} sec".format(duration)
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, current_message , (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, total_message , (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, duration_message , (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

            #Publish to MQTT Server
            client.publish("person", json.dumps({"count": p_counts}))

        ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            infer_network.clean()
    
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.loop_stop()
    client.disconnect()

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
