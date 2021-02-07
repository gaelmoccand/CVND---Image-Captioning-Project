# Image-Captioning-Project

In this project, a neural network architecture is used to automatically generate captions from images.

![image](https://github.com/gaelmoccand/Image-Captioning/blob/master/image-captioning.png)


After using the Microsoft Common Objects in COntext (MS COCO) dataset to train the network, new captions will be generated based on new images.

## The project Structure

1. ![model](Image-Captioning/model.py): containing the model architecture.
2. ![train](Image-Captioning/2_Training.ipynb): data pre-processing and training pipeline .
3. ![infer](Image-Captioning/3_Inference.ipynb): generate captions on test dataset using the trained model.


##  LSTMs

A very good summary on how LSTMs work can be found here ![lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Get Data

1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```
2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)
* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)


### References 

1. Show and Tell: A Neural Image Caption Generator [google](https://arxiv.org/pdf/1411.4555.pdf)
2. How to LSTM ![lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
