# Deception detection (Pytorch)
ICSSE 2020 : Face Expression and Tone of Voice for Deception System (Award of best student paper)

## Flowchart of deception detection
The overall flowchart for deception detection is illustrated below. We combine many features into a vector and then apply SVM to classification.
<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/Flowchart%20.png" width=50% height=50%>
</p>

## Face alignment
Because the face can be many angles, we need to align the face before using it.
<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/face%20alignment.png" width=50% height=50%>
</p>

## User instrctions
Our deception detection system comprises four parts：
1. 3D landmarks displacement
2. Emotion Unit
3. Action Unit
4. Emotion Audio unit

### Install Packages
Please see the ```requirements.txt``` for more details.

## Pre-trained models
- Please download the pre-trained models before you run the code
<https://drive.google.com/drive/folders/1A4_fAhWjDqzHLhLs4HHsErl90A1Wtgz9?usp=sharing>

## Dataset
- If you want access to the datasets below, please get in touch with the author who provided the datasets.
### Real-life trial dataset:
121 videos including 61 deceptive videos and 60 truth videos
### Bag-of-lies dataset:
325 videos including 162 deceptive videos and 163 truth videos
### MSPL-YTD dataset:
145 videos including 62 deceptive videos and 83 truth videos

## GUI demo
![image](https://github.com/come880412/Deception_detection/blob/main/img/Demo.png)
## Inference
```python=
python lie_GUI.py
```
- If you have any implementation problem, feel free to E-mail me! come880412@gmail.com
