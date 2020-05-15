# Project Write-Up
I am using this template provided with the project files for my writup.
Below are my findings and what I have done working on this project.

## Project Setup and Specifications
Here are the software and hardware specifications of my computer environment which was used to develope and test this People Counter App project;
- Hardware: HP ENVY Laptop 13-ad1xx 
            Procesor:Intel(R) Core(TM) i7-8550U CPU @ 180 GHz, 1992 MHz, 4 Cores, 8 Logical Processors
- Software: Operating System:   Windows 10 Enterprise 64 bit (Host)
            Ubuntu 16.04 LTS:   WSL on the Host Windows 10 OS
            Openvino Toolkit:   2019 R3
            Visual Studio Code: Version: 1.45.0 (user setup)
                                Electron: 7.2.4
                                Chrome: 78.0.3904.130
                                Node.js: 12.8.1
                                V8: 7.8.279.23-electron.0
                                OS: Windows_NT x64 10.0.18362
            Python:             Python 3.5
            

## Explaining Custom Layers
The Caffe model, which is the third model as per below I downloaded and used had the layers supported by the Openvino Model Optimizer. Hence there was no need to convert any custom layers. Only that I needed to use the cpu extension provided in OpenVino Toolkit to execute on my laptop computer.

However, if there were custom layers, I would need to follow the process behind converting custom layers as described in the openvino documentation for the specific framework. As for instance, for Caffe Models with custom layers, I have to do either the following;
      1) Register the custom layers as extensions to the Model Optimizer, or
      2) Register the custom layers as Custom and use the system Caffe to calculate the output shape of each custom Layer.
Or otherwise, use the heterogeneous plugin to run the inference model on multiple devices allowing the unsupported layers on one device to "fallback" to run on another device (e.g., CPU) that does support those layers. Unfortunately, I do not have the other hardwares at my disposal to do the testings.

Some of the potential reasons for handling custom layers is when a model is not supported by the model optimizer and or hardware issues where a hardware is not supported.

## Comparing Model Performance

My method(s) to compare the model performances was by using the Openvino Deep Learning Workbench and my computer Resource Monitor.

From the resource monitor the CPU usage increased from 10% to 30% and CPU minimum frequency increased from 70% to maximum frequency of 99% when executing the application using the model. So the CPU usage by the application and model was around 20%. 

To measure the model accuracy, I downloaded coco val2017 dataset and with its annotations and create configuration together with my model on the Deep Learning workbench. Below are the two different metric configuration results;
      1) Using COCO Precision
      - The model type is SSD for CPU targeted device and with maximum 20 deections, the accuracy was 0.63%
      2) Using Mean Average Precision (mAP)
      - The model type is SSD for CPU targeted device and with overlap threshold of 0.5, the accuracy was 3.61%
So depending on the type of metric configuration used, the model's accuracy was noticed to increase.

The size of the model was measured using ncdu 1.11. The original size and size after conversion are as per below;
            1) Before Conversion
             22.1 MiB - MobileNetSSD_deploy.caffemodel
             32.0 KiB - MobileNetSSD_deploy.prototxt
            2) After Conversion 
             22.1 MiB - MobileNetSSD_deploy.bin
             68.0 KiB - MobileNetSSD_deploy.xml
             16.0 KiB - MobileNetSSD_deploy.mapping                                                                                     
So it was noticed that the overal size of the produced files after conversion was a bit higher then the overal size of the original files by 52 KiB. That is inclding the .mappting file. There was no extraction of custom layers, otherwise, it would have been reduced dramatically.

And the inference time of the model was reduced after conversion. Refer below the Model Performance Summary extracted using Deep Learning Workbench.
            1) Baseline Summary with 1 Batch and 1 Parallel Stream
            Latency - 34.69 ms
            Throughput(FPS) - 26.81 
            2) Standard Inference with 1 Batch and 2 Parallel Stream
            Lateny - 43.71 ms
            Throughput(FPS) - 39.0
When executing the application on my laptop computer, the average inference time was around 30 ms.

Deploying at the edge has more advantages compared to deploying on the cloud in terms of network and costs. In cloud computing, devices send the visual data into the cloud for analysis, which then returns appropriate responses to the device for further action. This can lead to latencies in system response time and thus increases the network needs and can provide poor results. 

With respect to costs, deploying at the edge is significantly less compared to deploying at the cloud. Since computer vision is complex and powerful, it can become very expensive to build deploy and maintain. Thus cloud companies charge for inferencing per endpoint.This may be helpful for organizations that can pay on an ‘as needed’ basis; but becomes enormously burdensome for organizations that demand large amounts of real-time processing of videos/footages from online cameras in their organisations' premises.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are;
    1) School's attendance checker, 
    2) Construction sites' workers counter
    3) Workplace/Office Logger

Each of these use cases would be useful because the people counter app can be used to count and record the number of people in the each specific scenario. This can be used for safety and security purpose and improves business performances and productivity.

## Assess Effects on End User Needs
- Lighting:
  Light contributed much on the accurate detections of the model. I have noticed that lighting is much needed if appropriate detections are to be done. Hence, in a darker environment, we would need lighting installed or inbuilt into the detecting camera.
- Model accuracy:
  Model Accuracy is also an important aspect for the model to work efficiently. When converting to IR format, the inut and output paramters needed to be specified correctly to get maximum output and accuracy 
- Camera focal length/image size:
  Camera focal lenght/image sige can also affect the inferencing of the picture frames. Sometimes it can result in wrong object detection. Hence, this is also an important that needs to be addressed when setting a system for the end user.

## Model Research
After two attempts of researching a suitable model, I found the Caffe Model MobileNet SSD to be successfull. Below is the sucessful model with the parameters I used to convert to IR representations.
- Model: MobileNet SSD Caffe Model from Caffe Framework
- Model Source: Prototxt File and Caffemodel File:
  https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/people-counting/people-counting-opencv.zip
- The models are provided under the folder mobilenet_ssd in the zip folder as per the link.
- I converted the model to an Intermediate Representation with the following arguments...
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py 
  --input_model MobileNetSSD_deploy.caffemodel 
  --input_proto MobileNetSSD_deploy.prototxt  
  --mean_values [103.939,116.779,123.68] 
  --scale_values [127.5,127.5,127.5] 
  --reverse_input_channels

- This model was sufficient enough to run and output the expected results of my people counter app. Refer to the folder 'models/mobilenet_ssd' under the project folder for this model and the IR files

- Also note the changes I have made on the 'constants.js' in order to run on my local computer as per the README file instructions.

- I execute the application using the below commands after starting the Node server and UI and the FFMPEG server:

   python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/mobilenet_ssd/MobileNetSSD_deploy.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

- And to view the app, I access it on my local laptop via the address below on the browser:
  http://127.0.0.1:3000/

## Reference
- Most of the code/work are adapted from the my foundation course.
- Adrian at PyImage
  https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/people-counting/people-counting-opencv.zip
- Implement People Detection in Few Minutes:
  https://medium.com/@kurisutofusan/implement-people-detection-in-a-few-minutes-9da97e23dea8

