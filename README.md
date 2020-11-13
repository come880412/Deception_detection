# Deception detection
CVGIP2020:DECEPTION SYSTEM FOCUSING ON FACIAL EXPRESSION CHANGE WITH TONE OF VOICE(Award of best paper)\
ICSSE2020:Face Expression and Tone of Voice for Deception System(Award of best student paper)
# intro
There are four parts at our deception system：\
1.Emotion Unit\
2.Action Unit\
3.68 facial landmarks displacement\
4.Audio emotion feature\
We combine the above four features to do classification by SVM.And use this model to predict whether the subject is lie or not.
# Requirement
python = 3.6.10\
pytorch = 1.6.0\
pyqt = 5.12.3\
numpy = 1.18.5\
opencv = 4.3.0\
scikit-learn = 0.23.1\
joblib = 0.15.1\
imageio = 2.8.0\
skimage = 0.16.2
# Pretrained moedl
I have trained the four models <https://drive.google.com/drive/folders/1A4_fAhWjDqzHLhLs4HHsErl90A1Wtgz9?usp=sharing>
# Dataset
Real-life trail:\
121 videos including 61 deceptive videos and 60 truth videos\
Bag-of-lies:\
325 videos including 162 deceptive videos and 163 truth videos\
MSPL-YTD:\
145 videos including 62 deceptive videos and 83 truth videos
# GUI demo
![image](https://github.com/come880412/Deception_detection/blob/main/demo.jpg)
# Run
```python=
python lie_GUI.py
```
