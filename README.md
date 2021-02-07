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




### References 

1. Show and Tell: A Neural Image Caption Generator [google](https://arxiv.org/pdf/1411.4555.pdf)
2. How to LSTM ![lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
