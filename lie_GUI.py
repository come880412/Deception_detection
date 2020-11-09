import time
import cv2
import numpy as np
import sys
import os
import argparse
from random import sample
#Landmark load model
import yaml
#PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QTimer, QThread, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QBrush
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QGraphicsOpacityEffect
from PyQt5 import QtGui
import Qt_design as ui
from threading import Thread
#torch
import torch
import torchvision
#image transform
import skimage
import imageio
from skimage import img_as_ubyte
from model.Facedetection.config import device
#face detection
from model.Facedetection.utils import align_face, get_face_all_attributes, draw_bboxes
from model.Facedetection.RetinaFace.RetinaFaceDetection import retina_face
#Featrue extraction
import model.Emotion.lie_emotion_process as emotion
import model.action_v4_L12_BCE_MLSM.lie_action_process as action
from model.action_v4_L12_BCE_MLSM.config import Config
#Landmark
# from model.Landmark.TDDFA import TDDFA
# from model.Landmark.utils.render import render
# from model.Landmark.utils.functions import cv_draw_landmark, get_suffix
# save model
from joblib import dump, load

parser = argparse.ArgumentParser()
#Retina
parser.add_argument('--len_cut', default=30, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
#Landmark
parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d', '3d'])
# Emotion
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',help= '0 is self-attention; 1 is self + relation-attention')
parser.add_argument('--preTrain_path', '-pret', default='./model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar', type=str, help='pre-training model path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

#load model
Retina = retina_face(crop_size = 224, args = args) # Face detection
Emotion_class = emotion.Emotion_FAN(args = args)
Action_class = action.Action_Resnet(args= Config())
SVM_model = load('./model/SVM_model/se_res50+EU/split_svc_acc0.720_AUC0.828.joblib')
print('model is loaded')
class Landmark:
    def __init__(self,im,bbox,cfg,TDDFA,color):
        self.cfg = cfg
        self.tddfa = TDDFA
        self.boxes = bbox
        self.image = im
        self.color = color
        
    def main(self,index):
        dense_flag = args.opt in ('3d',)
        pre_ver = None
        self.boxes = [self.boxes[index]]
        param_lst, roi_box_lst = self.tddfa(self.image, self.boxes)
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        # refine
        param_lst, roi_box_lst = self.tddfa(self.image, [ver], crop_policy='landmark')
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        pre_ver = ver  # for tracking

        if args.opt == '2d':
            res = cv_draw_landmark(self.image, ver,color=self.color)
        elif args.opt == '3d':
            res = render(self.image, [ver])
        else:
            raise Exception(f'Unknown opt {args.opt}')
        
        lnd = ver.T
        # D1_i = np.sqrt(np.square(lnd[61][0]-lnd[67][0]) + np.square(lnd[61][1]-lnd[67][1]))
        # D1_o = np.sqrt(np.square(lnd[50][0]-lnd[58][0]) + np.square(lnd[50][1]-lnd[58][1]))
        # D2_i = np.sqrt(np.square(lnd[62][0]-lnd[66][0]) + np.square(lnd[62][1]-lnd[66][1]))
        # D2_o = np.sqrt(np.square(lnd[51][0]-lnd[57][0]) + np.square(lnd[51][1]-lnd[57][1]))
        # D3_i = np.sqrt(np.square(lnd[63][0]-lnd[65][0]) + np.square(lnd[63][1]-lnd[65][1]))
        # D3_o = np.sqrt(np.square(lnd[52][0]-lnd[56][0]) + np.square(lnd[52][1]-lnd[56][1]))
        res = res[int(roi_box_lst[0][1]):int(roi_box_lst[0][3]), int(roi_box_lst[0][0]):int(roi_box_lst[0][2])]
        # pm_ratio_1 = D1_i / D1_o
        # pm_ratio_2 = D2_i / D2_o
        # pm_ratio_3 = D3_i / D3_o
        # print('pm1:',pm_ratio_1)
        # print('pm2:',pm_ratio_2)
        # print('pm3:',pm_ratio_3)
        if res.shape[0] != 0 and res.shape[1] != 0:
            img_res = cv2.resize(res,(224,224))
        else:
            img_res = np.array([None])
        return img_res

#AU_pred thread
class AU_pred(QThread):
    trigger = pyqtSignal(list,list)
    def  __init__ (self,image):
        super(AU_pred ,self). __init__ ()
        self.face = image
    def run(self):
        logps, emb = Action_class._pred(self.face,Config)
        self.trigger.emit(emb.tolist(),logps.tolist())

class show(QThread):
    trigger = pyqtSignal(list,list,int)
    def  __init__ (self, frame_list ,frame_AU,log):
        super(show,self). __init__ ()
        self.frame_embed_list = frame_list
        self.frame_emb_AU = frame_AU
        self.log = log
    def pred(self):
        #Action calculation
        AU_list = self.log.tolist()[0]
        for index,i in enumerate(AU_list):
            if i >= 0.01:
                AU_list[index] = 1
            else:
                AU_list[index] = 0
        
        pred_score, self_embedding, relation_embedding = Emotion_class.validate(self.frame_embed_list) # Emotion_pred
        feature = np.concatenate((self.frame_emb_AU,relation_embedding.cpu().numpy()), axis = 1)
        results = SVM_model.predict(feature) # Lie_pred
        return AU_list, pred_score, results
    def run(self):
        logps,  pred_score, results  = self.pred()
        self.trigger.emit(logps,  pred_score.tolist(), results)
        

class lie_GUI(QDialog, ui.Ui_Dialog):
    def __init__(self, args):
        super(lie_GUI, self).__init__()
        print('Start deception detection')
        import qdarkstyle
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.mouth_count = 0
        self.frame_embed_list = [] # 儲存人臉
        self.frame_emb_AU = []
        self.log = []
        self.userface =[]
        self.color = (0, 255, 0)
        self.index = 0
        self.len_bbox = 1
        self.time = None 
        #Qt_design
        self.setupUi(self)
        self.Startlabel.setText('Press the button to upload a video or activate camera')
        self.Problem.setPlaceholderText('Enter the question')
        self.Record.setPlaceholderText('Enter the description')
        #hidden button
        self.Reset.setVisible(False)
        self.Finish.setVisible(False)
        self.truth_lie.setVisible(False)
        self.prob_label.setVisible(False)
        self.Start.setVisible(False)
        self.RecordStop.setVisible(False)
        self.filename.setVisible(False)
        self.videoprogress.setVisible(False)
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.Record_area.setVisible(False)
        self.Problem.setVisible(False)
        self.Record.setVisible(False)
        # self.Export.setVisible(False)
        self.camera_start.setVisible(False)
        self.Clear.setVisible(False)
        self.camera_finish.setVisible(False)
        #set style
        self.videoprogress.setStyleSheet("QProgressBar::chunk ""{""background-color: white;""}") ##4183c5
        #button click
        self.loadcamera.clicked.connect(self.start_webcam)
        self.loadvideo.clicked.connect(self.get_image_file)
        self.Reset.clicked.connect(self.Reset_but)
        self.Finish.clicked.connect(self.Reset_but)
        self.camera_finish.clicked.connect(self.Reset_but)
        self.Start.clicked.connect(self.time_start)
        self.RecordStop.clicked.connect(self.record_stop)
        self.camera_start.clicked.connect(self.Enter_problem)
        self.Clear.clicked.connect(self.cleartext)
        self.User0.clicked.connect(self.User_0)
        self.User1.clicked.connect(self.User_1)
        self.User2.clicked.connect(self.User_2)
        #button icon
        self.loadvideo.setIcon(QIcon('./icon/youtube.png')) # set button icon
        self.loadvideo.setIconSize(QSize(50,50)) # set icon size
        self.loadcamera.setIcon(QIcon('./icon/camera.png')) # set button icon
        self.loadcamera.setIconSize(QSize(50,50)) # set icon size
        self.Reset.setIcon(QIcon('./icon/reset.png')) # set button icon
        self.Reset.setIconSize(QSize(60,60)) # set icon size
        self.RecordStop.setIcon(QIcon('./icon/stop.png')) # set button icon
        self.RecordStop.setIconSize(QSize(30,30)) # set icon size
        #Landmark
        # self.cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        # self.tddfa = TDDFA(gpu_mode='gpu', **self.cfg)
        #攝像頭
        self.cap = None
        self.countframe = 0
        #timer
        self.timer = QTimer(self, interval=0)
        self.timer.timeout.connect(self.update_frame)

    def cleartext(self):
        self.Problem.clear()
        self.Record.clear()

    def Enter_problem(self):
        the_input = self.Problem.toPlainText() #文字框的字
        
        with open('Result.txt', 'a',newline='') as f:
            f.write("Problem :")
            f.write(the_input)
            
        self.time_start()

    def User_0(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 0
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)
        
    def User_1(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 1
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)

    def User_2(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 2
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)

    def time_start(self):
        if self.cap is not None:
            if self.mode == 'camera':
                self.Start.setVisible(False)
                self.RecordStop.setVisible(True)
            else:
                self.Start.setVisible(False)
                self.RecordStop.setVisible(True)
                self.videoprogress.setVisible(True)
            #把所有歸零
            self.User0.setVisible(False)
            self.User1.setVisible(False)
            self.User2.setVisible(False)
            self.camera_finish.setVisible(False)
            self.camera_start.setVisible(False)
            self.Clear.setVisible(False)
            self.prob_label.setVisible(False)
            self.Reset.setVisible(False)
            self.timer.start()
            self.truth_lie.setVisible(False)
            self.A01.setStyleSheet('''color:#c3c3c3''')
            self.A02.setStyleSheet('''color:#c3c3c3''')
            self.A04.setStyleSheet('''color:#c3c3c3''')
            self.A05.setStyleSheet('''color:#c3c3c3''')
            self.A06.setStyleSheet('''color:#c3c3c3''')
            self.A09.setStyleSheet('''color:#c3c3c3''')
            self.A12.setStyleSheet('''color:#c3c3c3''')
            self.A15.setStyleSheet('''color:#c3c3c3''')
            self.A17.setStyleSheet('''color:#c3c3c3''')
            self.A20.setStyleSheet('''color:#c3c3c3''')
            self.A25.setStyleSheet('''color:#c3c3c3''')
            self.A26.setStyleSheet('''color:#c3c3c3''')
            self.Happly_label.setStyleSheet('''color:#c3c3c3''')
            self.Angry_label.setStyleSheet('''color:#c3c3c3''')
            self.DIsgust_label.setStyleSheet('''color:#c3c3c3''')
            self.Fear_label.setStyleSheet('''color:#c3c3c3''')
            self.Sad_label.setStyleSheet('''color:#c3c3c3''')
            self.Neutral_label.setStyleSheet('''color:#c3c3c3''')
            self.Surprise_label.setStyleSheet('''color:#c3c3c3''')


    def record_stop(self):
        self.timer.stop()
        if self.mode =='video':
            self.RecordStop.setVisible(False)
            self.Start.setVisible(True)
            self.Finish.setVisible(True)
        else:
            self.frame_emb_AU = np.array(self.frame_emb_AU)
            self.frame_emb_AU = np.mean(self.frame_emb_AU, axis = 0)
            self.log = np.array(self.log)
            self.log = np.mean(self.log, axis = 0)
            self.show_thread = show(self.frame_embed_list, self.frame_emb_AU,self.log)
            self.show_thread.start()
            self.show_thread.trigger.connect(self.display_feature)
            self.frame_embed_list = []
            self.frame_emb_AU = []
            self.log = []
            self.camera_finish.setVisible(True)
            _translate = QtCore.QCoreApplication.translate
            self.camera_start.setText(_translate("Dialog", "Continue"))
            self.camera_start.setVisible(True)
            self.Clear.setVisible(True)
            self.RecordStop.setVisible(False)
            self.camera_finish.setVisible(True)

        self.Reset.setVisible(False)
        
        if self.lie_count >0 :
            lie_prob = round((self.lie_prob_count / self.lie_count) * 100)
            self.RecordStop.setVisible(False)
            self.prob_label.setVisible(True)
            self.prob_label.setText('The probability of deception: {:.0f}% '.format(lie_prob))

    def start_webcam(self):
        self.lie_count = 0
        self.lie_prob_count = 0
        self.loadvideo.setVisible(False)
        self.loadcamera.setVisible(False)
        # self.Start.setVisible(True)
        self.Reset.setVisible(True)
        if self.cap is None:
            self.Startlabel.setVisible(False)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.mode = 'camera'
            self.Problem.setVisible(True)
            self.Record.setVisible(True)
            self.Record_area.setVisible(True)

            self.Clear.setVisible(True)
            self.camera_start.setVisible(True)
        with open('Result.txt', 'w',newline='') as f:
            f.write("\t\t\t\t\t\tReport\n")
            

    def get_image_file(self):
        self.lie_count = 0
        self.lie_prob_count = 0
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Video File', r"<Default dir>", "Video files (*.mp4 *.avi)")
        if self.cap is None and file_name != '':
            self.Startlabel.setVisible(False)
            self.loadvideo.setVisible(False)
            self.loadcamera.setVisible(False)
            self.Start.setVisible(True)
            self.filename.setVisible(True)
            self.Finish.setVisible(False)
            self.Reset.setVisible(True)
            self.filename.setText('        Current file:\n{:^29}' .format(file_name.split('/')[-1]))
            self.cap = cv2.VideoCapture(file_name)
            self.frame_total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.mode = 'video'
    
    def face_recognition(self,bbox,img):
        self.Original.setPixmap(QPixmap(""))
        # self.Facealignment.setPixmap(QPixmap(""))
        # self.Landmark.setPixmap(QPixmap(""))

        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img_raw = img_raw.rgbSwapped()
        self.Original.setPixmap(QPixmap.fromImage(img_raw))

        self.Startlabel.setVisible(True)
        self.RecordStop.setVisible(False)

        if len(bbox) == 2:
            op0 = QGraphicsOpacityEffect()
            op0.setOpacity(0)
            op1 = QGraphicsOpacityEffect()
            op1.setOpacity(0)
            self.User0.setGraphicsEffect(op0)
            self.User1.setGraphicsEffect(op1)
            self.User0.setGeometry(QtCore.QRect(bbox[0][0], bbox[0][1]+25, 251, 91))
            self.User1.setGeometry(QtCore.QRect(bbox[1][0], bbox[1][1]+25, 251, 91))
            self.User0.setVisible(True)
            self.User1.setVisible(True)

        elif len(bbox) == 3:
            op0 = QGraphicsOpacityEffect()
            op0.setOpacity(0)
            op1 = QGraphicsOpacityEffect()
            op1.setOpacity(0)
            op2 = QGraphicsOpacityEffect()
            op2.setOpacity(0)
            self.User0.setGraphicsEffect(op0)
            self.User1.setGraphicsEffect(op1)
            self.User0.setGeometry(QtCore.QRect(bbox[0][0], bbox[0][1]+25, 251, 91))
            self.User1.setGeometry(QtCore.QRect(bbox[1][0], bbox[1][1]+25, 251, 91))
            self.User2.setGeometry(QtCore.QRect(bbox[2][0], bbox[2][1]+25, 251, 91))
            self.User0.setVisible(True)
            self.User1.setVisible(True)
            self.User2.setVisible(True)
        self.Startlabel.setText('Choose the user you want to detect!')
        
        

    def update_frame(self):
        ret, im = self.cap.read()
        im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA)
        self.im = im
        show_img = im.copy()
        self.countframe += 1
        #影片讀條
        if self.mode == 'video':
            self.videoprogress.setValue((round(self.countframe / self.frame_total ,2) * 100 ))

        image = skimage.img_as_float(im).astype(np.float32)
        frame = img_as_ubyte(image)
        self.img_raw, output_raw, output_points,bbox,self.face_list = Retina.detect_face(frame) # face detection
        #若只有一個臉，正常顯示
        if len(bbox) == 1:
            self.index = 0
            self.userface = self.face_list[self.index]
            self.len_bbox = 1
        elif len(bbox) >= 2:
            if len(self.userface):
                dist_list = []
                self.face_list = np.array(self.face_list)
                for i in range(len(bbox)):
                    dist = np.sqrt(np.sum(np.square(np.subtract(self.userface[:], self.face_list[i, :]))))
                    dist_list.append(dist)
                dist_list = np.array(dist_list)
                self.index = np.argmin(dist_list)

        if(len(output_points)):
            #face_align
            out_raw = align_face(output_raw, output_points[self.index], crop_size_h = 112, crop_size_w = 112)
            out_raw = cv2.resize(out_raw,(224, 224))
            #Landmark
            # _landmark = Landmark(im,bbox,self.cfg,self.tddfa,self.color)
            # landmark_img = _landmark.main(self.index)

            cv2.rectangle(show_img, (bbox[self.index][0], bbox[self.index][1]), (bbox[self.index][2], bbox[self.index][3]), (0, 0, 255), 2)
            self.face_align = out_raw
            self.bbox = bbox
            self.displayImage(show_img,bbox,True)
            self.frame_embed_list.append(out_raw) #儲存人臉
            #計算AU
            self.lnd_AU = AU_pred(out_raw)
            self.lnd_AU.start()
            self.lnd_AU.trigger.connect(self.AU_store)
        #沒有臉的時候
        else:
            self.frame_embed_list = []
            self.frame_emb_AU = []
            self.log = []
            self.A01.setStyleSheet('''color:#e8e8e8''')  # #e8e8e8
            self.A02.setStyleSheet('''color:#e8e8e8''')
            self.A04.setStyleSheet('''color:#e8e8e8''')
            self.A05.setStyleSheet('''color:#e8e8e8''')
            self.A06.setStyleSheet('''color:#e8e8e8''')
            self.A09.setStyleSheet('''color:#e8e8e8''')
            self.A12.setStyleSheet('''color:#e8e8e8''')
            self.A15.setStyleSheet('''color:#e8e8e8''')
            self.A17.setStyleSheet('''color:#e8e8e8''')
            self.A20.setStyleSheet('''color:#e8e8e8''')
            self.A25.setStyleSheet('''color:#e8e8e8''')
            self.A26.setStyleSheet('''color:#e8e8e8''')
            self.Happly_label.setStyleSheet('''color:#e8e8e8''')
            self.Angry_label.setStyleSheet('''color:#e8e8e8''')
            self.DIsgust_label.setStyleSheet('''color:#e8e8e8''')
            self.Fear_label.setStyleSheet('''color:#e8e8e8''')
            self.Sad_label.setStyleSheet('''color:#e8e8e8''')
            self.Neutral_label.setStyleSheet('''color:#e8e8e8''')
            self.Surprise_label.setStyleSheet('''color:#e8e8e8''')
            self.truth_lie.setVisible(False)
            self.displayImage(im, face_num = None)
            
    def AU_store(self,AU_emb,log):
        AU_emb = torch.FloatTensor(AU_emb)
        log = torch.FloatTensor(log)
        # print(log)
        self.frame_emb_AU.append(AU_emb.cpu().numpy())
        self.log.append(log.cpu().numpy())

    def displayImage(self, img,bbox=None,face_num = None ):
        #定義參數
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        #顯示影像
        if face_num:
            img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            img_raw = img_raw.rgbSwapped()
            # if lnd_img.any() != None:
            #     landmark_image = QImage(lnd_img, lnd_img.shape[1], lnd_img.shape[0], lnd_img.strides[0], qformat)
            #     landmark_image = landmark_image.rgbSwapped()
            #     self.Landmark.setPixmap(QPixmap.fromImage(landmark_image))

            # align_img = QImage(face_align, face_align.shape[1], face_align.shape[0], face_align.strides[0], qformat)
            # align_img = align_img.rgbSwapped()
            
            self.Original.setPixmap(QPixmap.fromImage(img_raw))
            # self.Facealignment.setPixmap(QPixmap.fromImage(align_img))
            
            #若大於兩個人，則選擇要哪個人
            if len(bbox) >=2 and self.len_bbox != len(bbox):
                self.frame_embed_list = []
                self.frame_emb_AU = []
                self.log = []
                self.len_bbox = len(bbox)
                self.timer.stop()
                self.face_recognition(bbox,self.img_raw)
            #若是影片，則len_cut禎計算一次結果
            if self.mode =='video':
                if len(self.frame_embed_list) == args.len_cut:
                    self.frame_emb_AU = np.array(self.frame_emb_AU)
                    self.frame_emb_AU = np.mean(self.frame_emb_AU, axis = 0)
                    self.log = np.array(self.log)
                    self.log = np.mean(self.log, axis = 0)
                    self.show_thread = show(self.frame_embed_list, self.frame_emb_AU,self.log)
                    self.show_thread.start()
                    self.show_thread.trigger.connect(self.display_feature)
                    self.frame_embed_list = []
                    self.frame_emb_AU = []
                    self.log = []

                
        else:
            img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            img_raw = img_raw.rgbSwapped()
            self.Original.setPixmap(QPixmap.fromImage(img_raw))
            # self.Facealignment.setPixmap(QPixmap(""))
            # self.Landmark.setPixmap(QPixmap(""))
            if self.mode =='camera' :
                    self.countframe =0

        if self.mode =='video' :
            if self.countframe == self.frame_total:
                self.prob_label.setVisible(True)
                self.RecordStop.setVisible(False)
                self.Reset.setVisible(True)
                if self.lie_count != 0:
                    lie_prob = round((self.lie_prob_count / self.lie_count) * 100)
                    self.prob_label.setText('The probability of deception: {:.0f}% '.format(lie_prob))
                self.timer.stop()
    def display_feature(self, logps, pred_score, results):
        pred_score = torch.Tensor(pred_score)
        #intialization
        self.A01.setStyleSheet('''color:#c3c3c3''')
        self.A02.setStyleSheet('''color:#c3c3c3''')
        self.A04.setStyleSheet('''color:#c3c3c3''')
        self.A05.setStyleSheet('''color:#c3c3c3''')
        self.A06.setStyleSheet('''color:#c3c3c3''')
        self.A09.setStyleSheet('''color:#c3c3c3''')
        self.A12.setStyleSheet('''color:#c3c3c3''')
        self.A15.setStyleSheet('''color:#c3c3c3''')
        self.A17.setStyleSheet('''color:#c3c3c3''')
        self.A20.setStyleSheet('''color:#c3c3c3''')
        self.A25.setStyleSheet('''color:#c3c3c3''')
        self.A26.setStyleSheet('''color:#c3c3c3''')
        self.Happly_label.setStyleSheet('''color:#c3c3c3''')
        self.Angry_label.setStyleSheet('''color:#c3c3c3''')
        self.DIsgust_label.setStyleSheet('''color:#c3c3c3''')
        self.Fear_label.setStyleSheet('''color:#c3c3c3''')
        self.Sad_label.setStyleSheet('''color:#c3c3c3''')
        self.Neutral_label.setStyleSheet('''color:#c3c3c3''')
        self.Surprise_label.setStyleSheet('''color:#c3c3c3''')
        self.truth_lie.setText('')

        if results ==1:
            self.color = (0, 0, 255) # red
            self.truth_lie.setText('Deception!')
            self.truth_lie.setStyleSheet('''QPushButton{background:#fff;border-radius:5px;color: red;}''')
            self.truth_lie.setVisible(True)
            self.lie_prob_count += 1
            #Emotion unit
            if pred_score.cpu().numpy().argmax() == 0:
                self.Happly_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 1:
                self.Angry_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 2:
                self.DIsgust_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 3:
                self.Fear_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 4:
                self.Sad_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 5:
                self.Neutral_label.setStyleSheet('''color:red''')
            elif pred_score.cpu().numpy().argmax() == 6:
                self.Surprise_label.setStyleSheet('''color:red''')
            #Action unit
            if logps[0]==1:
                self.A01.setStyleSheet('''color:red''')
            if logps[1]==1:
                self.A02.setStyleSheet('''color:red''')
            if logps[2]==1:
                self.A04.setStyleSheet('''color:red''')
            if logps[3]==1:
                self.A05.setStyleSheet('''color:red''')
            if logps[4]==1:
                self.A06.setStyleSheet('''color:red''')
            if logps[5]==1:
                self.A09.setStyleSheet('''color:red''')
            if logps[6]==1:
                self.A12.setStyleSheet('''color:red''')
            if logps[7]==1:
                self.A15.setStyleSheet('''color:red''')
            if logps[8]==1:
                self.A17.setStyleSheet('''color:red''')
            if logps[9]==1:
                self.A20.setStyleSheet('''color:red''')
            if logps[10]==1:
                self.A25.setStyleSheet('''color:red''')
            if logps[11]==1:
                self.A26.setStyleSheet('''color:red''')

            
        else:
            self.color = (0, 255, 0) #green
            self.truth_lie.setText('Truth!')
            self.truth_lie.setStyleSheet('''QPushButton{background:#fff;border-radius:5px;color: green;}''')
            self.truth_lie.setVisible(True)
            #Emotion unit
            if pred_score.cpu().numpy().argmax() == 0:
                self.Happly_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 1:
                self.Angry_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 2:
                self.DIsgust_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 3:
                self.Fear_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 4:
                self.Sad_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 5:
                self.Neutral_label.setStyleSheet('''color:green''')
            elif pred_score.cpu().numpy().argmax() == 6:
                self.Surprise_label.setStyleSheet('''color:green''')
            #Action unit
            if logps[0]==1:
                self.A01.setStyleSheet('''color:green''')
            if logps[1]==1:
                self.A02.setStyleSheet('''color:green''')
            if logps[2]==1:
                self.A04.setStyleSheet('''color:green''')
            if logps[3]==1:
                self.A05.setStyleSheet('''color:green''')
            if logps[4]==1:
                self.A06.setStyleSheet('''color:green''')
            if logps[5]==1:
                self.A09.setStyleSheet('''color:green''')
            if logps[6]==1:
                self.A12.setStyleSheet('''color:green''')
            if logps[7]==1:
                self.A15.setStyleSheet('''color:green''')
            if logps[8]==1:
                self.A17.setStyleSheet('''color:green''')
            if logps[9]==1:
                self.A20.setStyleSheet('''color:green''')
            if logps[10]==1:
                self.A25.setStyleSheet('''color:green''')
            if logps[11]==1:
                self.A26.setStyleSheet('''color:green''')

            
        self.frame_embed_list = []
        self.frame_emb_AU = []
        self.lie_count +=1
        if self.mode =='camera' :
            self.countframe =0
            with open('Result.txt', 'a',newline='') as f:

                f.write("\nEmotion unit:")
                if pred_score.cpu().numpy().argmax() == 0:
                    f.write('Happy')
                elif pred_score.cpu().numpy().argmax() == 1:
                    f.write("Angry")
                elif pred_score.cpu().numpy().argmax() == 2:
                    f.write("Disgust")
                elif pred_score.cpu().numpy().argmax() == 3:
                    f.write("Fear")
                elif pred_score.cpu().numpy().argmax() == 4:
                    f.write("Sad")
                elif pred_score.cpu().numpy().argmax() == 5:
                    f.write("Neutral")
                elif pred_score.cpu().numpy().argmax() == 6:
                    f.write("Surprise")

                f.write('\nAction unit:')
                if logps[0]==1:
                    f.write('Inner brow raiser\t')
                if logps[1]==1:
                    f.write('Outer brow raiser\t')
                if logps[2]==1:
                    f.write('Brow lower\t')
                if logps[3]==1:
                    f.write('Upper Lid Raiser\t')
                if logps[4]==1:
                    f.write('Cheek raiser\t')
                if logps[5]==1:
                    f.write('Nose wrinkle\t')
                if logps[6]==1:
                    f.write('Lip corner puller\t')
                if logps[7]==1:
                    f.write('Lip corner depressor\t')
                if logps[8]==1:
                    f.write('Chin raiser\t')
                if logps[9]==1:
                    f.write('Lip Stretcher\t')
                if logps[10]==1:
                    f.write('Lips part\t')
                if logps[11]==1:
                    f.write('Jaw drop\t')
                
                f.write('\nLie detection:')
                if results == 1:
                    f.write('Deception!')
                else:
                    f.write('Truth!')
                the_output = self.Record.toPlainText()
                f.write('\nDescription:')
                f.write(the_output)
                f.write('\n\n')

            # _landmark = Landmark(self.im,self.bbox,self.cfg,self.tddfa,self.color)
            # lnd_img = _landmark.main(self.index)
            qformat = QImage.Format_Indexed8
            if len(self.im.shape)==3 :
                if self.im.shape[2]==4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            # if lnd_img.any() != None:
            #     landmark_image = QImage(lnd_img, lnd_img.shape[1], lnd_img.shape[0], lnd_img.strides[0], qformat)
            #     landmark_image = landmark_image.rgbSwapped()
            #     self.Landmark.setPixmap(QPixmap.fromImage(landmark_image))
    def Reset_but(self):
        self.Reset.setVisible(False)
        self.camera_finish.setVisible(False)
        self.Finish.setVisible(False)
        self.truth_lie.setVisible(False)
        self.videoprogress.setVisible(False)
        self.filename.setVisible(False)
        self.loadcamera.setVisible(True)
        self.loadvideo.setVisible(True)
        self.Startlabel.setVisible(True)
        self.Start.setVisible(False)
        self.prob_label.setVisible(False)
        self.Problem.setVisible(False)
        self.Record_area.setVisible(False)
        self.Record.setVisible(False)
        self.camera_start.setVisible(False)
        self.Clear.setVisible(False)
        _translate = QtCore.QCoreApplication.translate
        self.camera_start.setText(_translate("Dialog", "Start"))
        self.Problem.clear() 
        self.Record.clear()
        self.A01.setStyleSheet('''color:#c3c3c3''')
        self.A02.setStyleSheet('''color:#c3c3c3''')
        self.A04.setStyleSheet('''color:#c3c3c3''')
        self.A05.setStyleSheet('''color:#c3c3c3''')
        self.A06.setStyleSheet('''color:#c3c3c3''')
        self.A09.setStyleSheet('''color:#c3c3c3''')
        self.A12.setStyleSheet('''color:#c3c3c3''')
        self.A15.setStyleSheet('''color:#c3c3c3''')
        self.A17.setStyleSheet('''color:#c3c3c3''')
        self.A20.setStyleSheet('''color:#c3c3c3''')
        self.A25.setStyleSheet('''color:#c3c3c3''')
        self.A26.setStyleSheet('''color:#c3c3c3''')
        self.Happly_label.setStyleSheet('''color:#c3c3c3''')
        self.Angry_label.setStyleSheet('''color:#c3c3c3''')
        self.DIsgust_label.setStyleSheet('''color:#c3c3c3''')
        self.Fear_label.setStyleSheet('''color:#c3c3c3''')
        self.Sad_label.setStyleSheet('''color:#c3c3c3''')
        self.Neutral_label.setStyleSheet('''color:#c3c3c3''')
        self.Surprise_label.setStyleSheet('''color:#c3c3c3''')
        self.color = (0, 255, 0)
        self.truth_lie.setText('Lie_truth')
        # self.truth_lie.setStyleSheet('''QPushButton{background:##ff70ff;border-radius:5px;}''')
        self.videoprogress.setValue(0)
        self.frame_embed_list = []
        self.frame_emb_AU = []
        self.userface = []
        self.countframe = 0
        self.index = 0
        self.len_bbox = 1
        self.Original.setPixmap(QPixmap(""))
        # self.Facealignment.setPixmap(QPixmap(""))
        # self.Landmark.setPixmap(QPixmap(""))
        self.filename.setText('')
        self.Startlabel.setText('Press the button to upload a video or activate camera')
        # self.Facedetection.setPixmap(QPixmap(""))
        self.timer.stop()
        if self.cap != None:
            self.cap.release()
            self.cap = None
if __name__=='__main__':
    app = QApplication(sys.argv)
    window = lie_GUI(args)
    window.show()
    sys.exit(app.exec_())