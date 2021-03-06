# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program


## Introduction
The goal of this project is to build a Fully Convolutional Network (FCN) and use it to label the pixels of road images.

## Setup

### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

### Frameworks and Packages

Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

### Dataset

Dataset is available for download at [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## Implement

The implementation of the code in the [main.py](./main.py) module was based on the lessons, the [Project Q&A](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/595f35e6-b940-400f-afb2-2015319aa640/lessons/1b046c47-76e3-45de-8be7-8bc6b4361b18/concepts/2a478851-eebb-47c9-acc3-c6a92f1f3e61) and information from the article [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer and Trevor Darrell from UC Berkeley.

The main function is the function layers in [main.py](./main.py) (from line 52 to line 98).

The function builds a decoder for the FCN starting from the VGG's layers 7.

It first creates a 1x1 convolutional layer from the VGG layers 7 (line 62).

It then up-scales (transposes) the subsequent convolutional layers by 2, 2 and 8 to get the original dimension of the input image (lines 67, 79 and 91).

In between upscaling the decoder uses VGG's layers 4 and 3 to create skip connections (lines 72 to 76 and lines 84 to 88)

The optimize function (line 101 to 126) uses Adam optimizer for training and minimizing of loss function.


## Run
Run the following command to run the project:


```
python main.py
```

The tests were run with Udacity GPU workspace.

The best results were obtained by training the FCN with the following hyperparameters:

Epochs: 48

Batch size: 10

Keep-prob: 0.5

Learning rate: 0.00085


## Example Outputs

Here are some examples of output from a trained network:

![img1](./runs/1545048500.3815806/uu_000086.png)

![img2](./runs/1545048500.3815806/uu_000085.png)

![img3](./runs/1545048500.3815806/uu_000032.png)

![img4](./runs/1545048500.3815806/uu_000044.png)

![img5](./runs/1545048500.3815806/uu_000057.png)

![img6](./runs/1545048500.3815806/uu_000061.png)


## [The Rubric](https://review.udacity.com/#!/rubrics/989/view)


The function load_vgg was implemented from line 23 to line 49 in [main.py](./main.py). 

The function layers was implemented from line 52 to line 98 in [main.py](./main.py).

The function optimize was implemented from line 101 to line 126 in [main.py](./main.py).

The function train_nn was implemented from line 129 to line 175 in [main.py](./main.py). 

The above functions passed the test in [main.py](./main.py).

The average, minimum, maximum and standard deviation of the loss of each epoch of the network were printed out while the network is training.

The following is a fragment from the [run log](./runs/run_log_48_10_00085.log), which shows the function tests and the output of each epoch from the network while training.
 
```

Testing load_vgg
2018-12-17 11:10:07.912483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
Tests Passed
Testing layers function
Tests Passed
Testing optimize function
2018-12-17 11:10:15.758778: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
Tests Passed
Testing training function
2018-12-17 11:10:17.229849: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
Tests Passed
Testing kitti_dataset
Tests Passed
2018-12-17 11:10:17.257801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
Start training, Epochs:  48 , Batch size:  10 , Keep prob:  0.5 , Learning rate:  0.00085
Epoch,  1 , Run time:  84.283367395401 , Mean loss,  0.958544 , Min loss,  0.34525 , Max loss,  9.93832 , Stdev,  1.72157
Epoch,  2 , Run time:  71.9490327835083 , Mean loss,  0.272717 , Min loss,  0.170564 , Max loss,  0.4097 , Stdev,  0.0544641
Epoch,  3 , Run time:  71.77907943725586 , Mean loss,  0.182951 , Min loss,  0.144888 , Max loss,  0.240852 , Stdev,  0.0242299
Epoch,  4 , Run time:  71.91668343544006 , Mean loss,  0.167156 , Min loss,  0.133189 , Max loss,  0.210576 , Stdev,  0.0179439
Epoch,  5 , Run time:  71.84802436828613 , Mean loss,  0.151692 , Min loss,  0.124436 , Max loss,  0.186558 , Stdev,  0.014765

```

It was found that the best results were obtained by training the network with the following values of hyperparameters:
```  
Start training, Epochs:  48 , Batch size:  10 , Keep prob:  0.5 , Learning rate:  0.00085
```

While the network is training, the averaged loss was decreasing over time as in the following:

```

Epoch,  1 , Run time:  84.283367395401 , Mean loss,  0.958544 , Min loss,  0.34525 , Max loss,  9.93832 , Stdev,  1.72157
Epoch,  2 , Run time:  71.9490327835083 , Mean loss,  0.272717 , Min loss,  0.170564 , Max loss,  0.4097 , Stdev,  0.0544641
Epoch,  3 , Run time:  71.77907943725586 , Mean loss,  0.182951 , Min loss,  0.144888 , Max loss,  0.240852 , Stdev,  0.0242299
Epoch,  4 , Run time:  71.91668343544006 , Mean loss,  0.167156 , Min loss,  0.133189 , Max loss,  0.210576 , Stdev,  0.0179439
Epoch,  5 , Run time:  71.84802436828613 , Mean loss,  0.151692 , Min loss,  0.124436 , Max loss,  0.186558 , Stdev,  0.014765
Epoch,  6 , Run time:  72.03599977493286 , Mean loss,  0.146156 , Min loss,  0.111325 , Max loss,  0.18377 , Stdev,  0.0177905
Epoch,  7 , Run time:  71.80563855171204 , Mean loss,  0.146286 , Min loss,  0.103637 , Max loss,  0.20188 , Stdev,  0.0242049
Epoch,  8 , Run time:  71.91143131256104 , Mean loss,  0.131833 , Min loss,  0.0901057 , Max loss,  0.165762 , Stdev,  0.0175261
Epoch,  9 , Run time:  71.90694284439087 , Mean loss,  0.126162 , Min loss,  0.0821893 , Max loss,  0.168468 , Stdev,  0.0206832
Epoch,  10 , Run time:  71.84077382087708 , Mean loss,  0.12168 , Min loss,  0.0879852 , Max loss,  0.168633 , Stdev,  0.0166663
Epoch,  11 , Run time:  71.82684707641602 , Mean loss,  0.112289 , Min loss,  0.0821214 , Max loss,  0.157662 , Stdev,  0.0160633
Epoch,  12 , Run time:  71.84050369262695 , Mean loss,  0.106062 , Min loss,  0.0856564 , Max loss,  0.127121 , Stdev,  0.0110473
Epoch,  13 , Run time:  71.826979637146 , Mean loss,  0.101202 , Min loss,  0.0639056 , Max loss,  0.20031 , Stdev,  0.0246669
Epoch,  14 , Run time:  72.04916143417358 , Mean loss,  0.0865251 , Min loss,  0.0607056 , Max loss,  0.117045 , Stdev,  0.0147749
Epoch,  15 , Run time:  72.06934022903442 , Mean loss,  0.0811523 , Min loss,  0.0568748 , Max loss,  0.112085 , Stdev,  0.0135938
Epoch,  16 , Run time:  71.95822381973267 , Mean loss,  0.0796457 , Min loss,  0.0476372 , Max loss,  0.128039 , Stdev,  0.0165744
Epoch,  17 , Run time:  72.0634982585907 , Mean loss,  0.0703683 , Min loss,  0.0521532 , Max loss,  0.0909552 , Stdev,  0.0101762
Epoch,  18 , Run time:  72.10915565490723 , Mean loss,  0.0702743 , Min loss,  0.0527268 , Max loss,  0.101339 , Stdev,  0.0125853
Epoch,  19 , Run time:  72.00783658027649 , Mean loss,  0.065973 , Min loss,  0.049947 , Max loss,  0.0839099 , Stdev,  0.00935982
Epoch,  20 , Run time:  72.01335501670837 , Mean loss,  0.0608548 , Min loss,  0.0413715 , Max loss,  0.0782774 , Stdev,  0.00861834
Epoch,  21 , Run time:  72.14358353614807 , Mean loss,  0.0580221 , Min loss,  0.041287 , Max loss,  0.0913076 , Stdev,  0.0109748
Epoch,  22 , Run time:  72.15262508392334 , Mean loss,  0.05601 , Min loss,  0.0364977 , Max loss,  0.0916767 , Stdev,  0.0125256
Epoch,  23 , Run time:  72.09018135070801 , Mean loss,  0.0537182 , Min loss,  0.031197 , Max loss,  0.0694403 , Stdev,  0.00855067
Epoch,  24 , Run time:  72.11547207832336 , Mean loss,  0.0542129 , Min loss,  0.0405777 , Max loss,  0.0750362 , Stdev,  0.00779797
Epoch,  25 , Run time:  71.94964575767517 , Mean loss,  0.0516125 , Min loss,  0.0359205 , Max loss,  0.0743462 , Stdev,  0.0101681
Epoch,  26 , Run time:  71.9767689704895 , Mean loss,  0.0471109 , Min loss,  0.0347839 , Max loss,  0.0689871 , Stdev,  0.0090549
Epoch,  27 , Run time:  72.10543084144592 , Mean loss,  0.0447665 , Min loss,  0.0289686 , Max loss,  0.0625283 , Stdev,  0.00777084
Epoch,  28 , Run time:  72.08963918685913 , Mean loss,  0.0475692 , Min loss,  0.0301632 , Max loss,  0.0690742 , Stdev,  0.0100149
Epoch,  29 , Run time:  71.9238977432251 , Mean loss,  0.0538326 , Min loss,  0.0340213 , Max loss,  0.0892068 , Stdev,  0.0144635
Epoch,  30 , Run time:  72.10965037345886 , Mean loss,  0.0477501 , Min loss,  0.034183 , Max loss,  0.0625383 , Stdev,  0.00704227
Epoch,  31 , Run time:  72.256840467453 , Mean loss,  0.0444718 , Min loss,  0.0281866 , Max loss,  0.0545788 , Stdev,  0.00712883
Epoch,  32 , Run time:  72.1117205619812 , Mean loss,  0.0422397 , Min loss,  0.0324779 , Max loss,  0.0568457 , Stdev,  0.00703022
Epoch,  33 , Run time:  71.97964406013489 , Mean loss,  0.0410902 , Min loss,  0.0249044 , Max loss,  0.0627226 , Stdev,  0.00812341
Epoch,  34 , Run time:  72.18597221374512 , Mean loss,  0.0392228 , Min loss,  0.028436 , Max loss,  0.055015 , Stdev,  0.00679633
Epoch,  35 , Run time:  71.9233021736145 , Mean loss,  0.0381524 , Min loss,  0.0292824 , Max loss,  0.0507232 , Stdev,  0.00627963
Epoch,  36 , Run time:  72.06767225265503 , Mean loss,  0.0353046 , Min loss,  0.0232156 , Max loss,  0.0520525 , Stdev,  0.00599999
Epoch,  37 , Run time:  72.04632496833801 , Mean loss,  0.0345983 , Min loss,  0.0242703 , Max loss,  0.0471591 , Stdev,  0.00599739
Epoch,  38 , Run time:  71.99161052703857 , Mean loss,  0.0340329 , Min loss,  0.0227281 , Max loss,  0.0471781 , Stdev,  0.00543071
Epoch,  39 , Run time:  72.11951541900635 , Mean loss,  0.0333509 , Min loss,  0.0233362 , Max loss,  0.0459563 , Stdev,  0.0055671
Epoch,  40 , Run time:  72.17704033851624 , Mean loss,  0.0325103 , Min loss,  0.0211155 , Max loss,  0.0480216 , Stdev,  0.00552052
Epoch,  41 , Run time:  72.1227457523346 , Mean loss,  0.0316856 , Min loss,  0.0239942 , Max loss,  0.0438205 , Stdev,  0.0043315
Epoch,  42 , Run time:  72.08833026885986 , Mean loss,  0.0300436 , Min loss,  0.0191601 , Max loss,  0.0400367 , Stdev,  0.00503935
Epoch,  43 , Run time:  72.08342623710632 , Mean loss,  0.0294937 , Min loss,  0.0229533 , Max loss,  0.0394148 , Stdev,  0.00355553
Epoch,  44 , Run time:  72.10313630104065 , Mean loss,  0.0290065 , Min loss,  0.0211092 , Max loss,  0.0349486 , Stdev,  0.00320833
Epoch,  45 , Run time:  72.123774766922 , Mean loss,  0.0280983 , Min loss,  0.0198009 , Max loss,  0.0372108 , Stdev,  0.00404395
Epoch,  46 , Run time:  72.07649445533752 , Mean loss,  0.0276349 , Min loss,  0.0165841 , Max loss,  0.0384665 , Stdev,  0.00456424
Epoch,  47 , Run time:  72.10004353523254 , Mean loss,  0.0270966 , Min loss,  0.0204851 , Max loss,  0.0363231 , Stdev,  0.00438135
Epoch,  48 , Run time:  72.0740852355957 , Mean loss,  0.0292163 , Min loss,  0.0214704 , Max loss,  0.0438521 , Stdev,  0.00482041
Training completed in  3469.3468902111053
Training Finished. Saving test images to: ./runs/1545048500.3815806
```

The network seemed to label the majority of pixels of road images correctly and met the requirement of the project. 


