#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.inference_req = None

    def load_model(self, model, device, cpu_extension):
        ### TODO: Load the model ###
        model_xml_file = model
        model_bin_file = os.path.splitext(model_xml_file)[0] + ".bin"

        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml_file, model_bin_file))
        self.network = IENetwork(model=model_xml_file, weights=model_bin_file)

        # Initialize the plugin
        log.info("Creating Inference Engine...")
        self.plugin = IECore()

        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(self.network, device)
        if len(supported_layers) == 0:
            log.error("There are no supported layers for the plugin for the specified device {}: ".
                    format(device))
            sys.exit(1)

        ### TODO: Add any necessary extensions ###
        # Using CPU so use CPU extension
        if cpu_extension and "CPU" in device:
            log.info("Adding extension: \n\t\{}".format(cpu_extension))
            self.plugin.add_extension(cpu_extension, device)
        
        ### TODO: Return the loaded inference plugin ###s
        self.exec_network = self.plugin.load_network(self.network, device)
        # Get the layers
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shape = self.network.inputs[self.input_blob].shape
        return input_shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id=0,inputs={self.input_blob: image})

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        output_shape = self.exec_network.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return output_shape
