# Deception_detection intro
There are four parts at our deception system：\
1.Emotion Unit\
2.Action Unit\
3.68 facial landmarks displacement\
4.Audio emotion feature\
\
We combine the above four features to do classification by SVM.And use this model to predict whether the subject is lie or not.
# Requirement
pytorch\
pyqt5\
argparse\
Numpy\
Opencv\
scikit-learn\
joblib\
imageio\
skimage
# Dataset
Real-life trail:\
121 videos including 61 deceptive videos and 60 truth videos\
Bag-of-lies:\
325 videos including 162 deceptive videos and 163 truth videos\
MSPL-YTD:\
145 videos including 62 deceptive videos and 83 truth videos
# GUI demo
![image](https://github.com/come880412/Deception_detection/blob/main/demo.jpg)
