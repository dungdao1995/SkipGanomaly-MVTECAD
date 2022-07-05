# SkipGANomaly-MVTECAD

## SkipGANomaly Model

In recent years, the infrastructure anomaly detection problems using the super- vised learning algorithms do not meet the demands because of unpredictability in acquiring various abnormal samples as well as the balance between the nor- mal dataset and the abnormal dataset. Currently, there is a tendency to use unsupervised learning algorithms in solving anomaly detection, and the stand- out solution is using generative models.

SkipGANomaly is a form of generative model and it is used widely in anomaly detection

### Overview of SkipGANomaly

<img width="409" alt="Screen Shot 2022-07-06 at 1 11 05 AM" src="https://user-images.githubusercontent.com/53828158/177432070-ac38bfc7-dad3-4c7c-bde9-c102935653b6.png">

### Detail of SkipGANomaly architecture

<img width="812" alt="Screen Shot 2022-07-06 at 1 11 24 AM" src="https://user-images.githubusercontent.com/53828158/177432129-1189a761-d21b-4001-971b-cf05c82f6b1a.png">

## Problem Definition

We have the large training dataset D containing M samples normal data D = X1,..Xm with the smaller testing dataset D′ has N samples including both nor- mal and abnormal data D′ = (X1, y1), ..(Xn, yn) with y belongs to [0,1]. 0 is the label of normal data and 1 is the label of abnormal data. In most cases, the number of data in training dataset is much larger than the number of the testing dataset so M >> N.

The first goal is modeling D to learn its manifold, and then detect the abnor- mal data in D′ as the outlier. During the training phase, the model learns the distribution of normal data and minimizes the **anomaly score A(x)**. We can use the thresh hold φ to detect the abnormal input data. **When we feed the abnormal data x′ to the model, the anomaly of x′ is A(x′) will higher than the threshold φ or A(x′) > φ**.


## THE MVTEC ANOMALY DETECTION DATASET (MVTEC AD)

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.
In this project we use SCREW dataset for this anomaly detection problem.

### Normal image and  Abnormal image

<img width="256" alt="Screen Shot 2022-03-03 at 9 13 15 PM" src="https://user-images.githubusercontent.com/53828158/177432312-b44f9126-77cd-4090-9021-87b059cf3e3d.png"> <img width="256" alt="Screen Shot 2022-03-03 at 9 13 15 PM" src="https://user-images.githubusercontent.com/53828158/177432333-84aa5ba7-923a-417c-9a54-52456c625804.png">

## Result

### Anomaly Score

<img width="1024" alt="Screen Shot 2022-07-06 at 1 25 55 AM" src="https://user-images.githubusercontent.com/53828158/177433687-95fd0273-609e-49d7-baee-ae1c9a4192c2.png">

### ROC and AUC

<img width="1024" alt="Screen Shot 2022-07-06 at 1 26 08 AM" src="https://user-images.githubusercontent.com/53828158/177433725-9aa9a6e2-0a94-42ba-8e63-e926b5edd589.png">


