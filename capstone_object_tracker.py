import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#arduino control
from pyfirmata import Arduino

#twilio Package
from twilio.rest import Client

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')



#(클래스선언)
class Person_status:
    # __init__ 초기화 메소드 (생성자) self == this
    def __init__(self, s_class_name, s_track_id, s_bx1, s_by1, s_bx2 ,s_by2, s_center_x, s_center_y, action = "None", action_change = 0, s_color = (0,255,0), pd_exist = -1):
        self.s_class_name = s_class_name
        self.s_track_id = s_track_id
        self.s_bx1 = s_bx1
        self.s_by1 = s_by1
        self.s_bx2 = s_bx2
        self.s_by2 = s_by2
        self.s_center_x = s_center_x
        self.s_center_y = s_center_y
        self.action = action         
        self.action_change = action_change
        self.s_color = s_color
        self.pd_exist = pd_exist
    
    def display(self):
        print("this is person status")
        print("class_name! : "+self.s_class_name+" ID : "+self.s_track_id)
        print("x1 : "+self.s_bx1 + " y1 : "+self.s_by1 + " x2 : "+self.s_bx2+" y2 : "+self.s_by2)
        print("---------------------------------------------------------------------------------------------------------")

#p1 = Object_status("aaa",1,2,3,4,5,"moving",1,20) ....  p1.

class Product_status:
    def __init__(self, s_class_name, s_track_id, s_bx1, s_by1, s_bx2, s_by2, s_center_x, s_center_y, action = "-", is_paid = 0, s_color = (0,0,255)):
        self.s_class_name = s_class_name
        self.s_track_id = s_track_id
        self.s_bx1 = s_bx1
        self.s_by1 = s_by1
        self.s_bx2 = s_bx2
        self.s_by2 = s_by2
        self.s_center_x = s_center_x
        self.s_center_y = s_center_y
        self.action = action
        self.is_paid = is_paid
        self.s_color = s_color

    def display(self):
        print("this is object status")
        print("class_name! : "+self.s_class_name+" ID : "+self.s_track_id)
        print("x1 : "+self.s_bx1 + " y1 : "+self.s_by1 + " x2 : "+self.s_bx2+" y2 : "+self.s_by2)
        print("action : " +self.action+" is_paid : "+self.is_paid)
        print("---------------------------------------------------------------------------------------------------------")



#holds 클래스를 만들어서 잡았던 사람의 아이디와 잡혔던 물건의 아이디를 넣어주려고했는데 클래스여서 list index 사용불가, len(holds) 하니까 값이 계속 늘어나서 사용불가 그냥 따로따로 만듬. 그리고 holded_pd 는 지워지면 다음프레임에서 hide 를 감지 못함 
#age 가 어느정도 지났다 싶으면 지워줘야될듯하다.

"""
class Holds_list:
    def __init__(self, person_id, product_id):
        self.person_id = person_id
        self.product_id = product_id
"""

#arduino set
comport = Arduino('COM4')
pin4 = comport.get_pin('d:4:o')
pin5 = comport.get_pin('d:5:o')
pin6 = comport.get_pin('d:6:o')

#twilio API
account_sid = 'ACc48fed17cfe71030681e5fb2b471ba7f'
auth_token = '7e2c73cfcd0cb950a413c66db27a9767'
client = Client(account_sid, auth_token)

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
        #vid = cv2.VideoCapture(1)
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

   
    # add code (구조체 선언)
    frame_num = 0
    #구조체 동적할당을위한 리스트 선언__현재프레임의 모든 오브젝트들이 들어간다. li = [] or li = list()
    #잡았던 사람의 track_id, 잡혔던 물건의 track_id
    hold_person = []
    holded_pd = []

    #사람클래스, 물건클래스
    Person_list = []
    Product_list = []

    #문자 전송
    send_count = 0

    #문의 좌표, 중앙좌표
    door_x1 = 495
    door_y1 = 116
    door_x2 = 605
    door_y2 = 288
                    #(x1  +  x2)/2 
    door_centerx = int((door_x1 + door_x2)/2)
                    #(y1  +  y2)/2
    door_centery = int((door_y1 + door_y2)/2)

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #원하는 클래스만 탐지
        #allowed_classes = list(class_names.values())
        allowed_classes = ['person', 'bottle', 'knife']
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        #colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        #add Code --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
       
        #현재 프레임의 오브젝트 갯수 구하기
        p_count = 0 #사람 수
        o_count = 0 #물건 수

        #문과 사람사이의 거리
        ptod_distance = 0
        
        for track in tracker.tracks: 
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #get status

            # x1,y1, x2,y2
            bbox = track.to_tlbr()
            # class_name
            class_name = track.get_class()
            # y1 + y2 /2  center_x (int)
            center_x = int((int(bbox[0]) + int(bbox[2]))/2)
            # x1 + x2 /2  center_y (int)
            center_y = int((int(bbox[1]) + int(bbox[3]))/2) 

            #insert object status
            if(int(num_objects) > 0):  #탐지된게 존재한다면
                if(str(class_name)=="person"):  #사람일때
                    p_count = p_count + 1
                    #Person_list.append(Person_status(class_name,int(track.track_id),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))
                    Person_list.append(Person_status(class_name,int(track.track_id),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))
                elif (str(class_name)=="bottle"):   #물건일때
                    o_count = o_count + 1
                    #Product_list.append(Product_status(class_name,int(track.track_id),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))
                    Product_list.append(Product_status("product",int(track.track_id),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))
                elif (str(class_name)=="knife"):   #물건일때
                    o_count = o_count + 1
                    #Product_list.append(Product_status(class_name,int(track.track_id),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))
                    Product_list.append(Product_status(class_name,200,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),center_x,center_y))

        #person 을 주체로 가지고 있는지 판별 (action = hold_product)
        for i in range(0, p_count) :
            for j in range(0, o_count):
                if(str(Product_list[j].s_class_name) == "product"):
                    if(Person_list[i].s_bx1 - 35 <= Product_list[j].s_center_x <= Person_list[i].s_bx2 + 35 and Person_list[i].s_by1 + 35 <= Product_list[j].s_center_y <= Person_list[i].s_by2 - 35):

                        Person_list[i].action = "hold_product"
                        Person_list[i].s_color = (255,127,0)
                        
                        holded_pd.append(Product_list[j].s_track_id) #잡혔던 물건 의 정보 저장 잡혔던 물건은 현재 물건의 트랙id 와 잡았던 사람의 트랙id 를 갖고가야한다.
                        holded_pd = list(set(holded_pd))

                        hold_person.append(Person_list[i].s_track_id) #물건을 잡았던 사람들 저장
                        hold_person = list(set(hold_person))
                
                if(str(Product_list[j].s_class_name) == "knife"):
                    Person_list[i].s_class_name = "robber"
                    Person_list[i].action = "threat"
                    Person_list[i].s_color = (0,0,0)
                    if send_count == 0 :
                        message = client.messages \
                                        .create(
                                            body="강도가 침입하였습니다!",
                                            from_='+12513125304',
                                            to='+821077452996'
                                        )
                        
                        send_count = 1
                        print(message.sid)


        #만약 사람이 hold 에서 none 으로 바뀐 경우
        if hold_person:
            ex_count = 0
            for i in range (0, p_count):
                for j in range(0, len(hold_person)):
                    if(int(hold_person[j]) == Person_list[i].s_track_id):
                        if( str(Person_list[i].action) == "None" ):
                            #print(str(Person_list[j].s_track_id) + "번 사람의 행동이 hold 에서 none 으로 바뀌었다") # hold_product 에서 none 으로 상태천이 되었을때
                            
                            #화면에 물체가 아무것도 없거나
                            if(o_count == 0):
                                Person_list[i].pd_exist = 0
                                break
                            else:
                                #물체가 있는데 다른 번호 일경우, 안될경우 
                                """
                                for k in range (0, o_count):                        
                                    try:
                                        num = holded_pd.index((Product_list[k].s_track_id))
                                        if(num >= 0):
                                            Person_list[i].pd_exist = 1             #현재프레임안에 잡혔던 물건이 존재. 즉 집었던 물건을 다시 내려놓음 holded_pd 에 해당하는 값이있으면 그 값이 있는 index 번호를 반환
                                        
                                    except ValueError:
                                        Person_list[i].pd_exist = 0                #holded_pd 안에 같은 값이 존재하지않음 : holded_pd.index 에서 값이없으니 Value Error 발생. 즉 집었던 물건을 숨겼을 가능성이 있음
                                """
                                
                                for k in range (0, o_count):
                                    for l in range (0, len(holded_pd)):
                                        if(Product_list[k].s_track_id == holded_pd[l]):
                                            ex_count = ex_count + 1
                                            Person_list[i].pd_exist = 1
                                            break
                                
                                if(ex_count == 0):
                                    Person_list[i].pd_exist = 0

                                                                        
      
        # action = hide_product로 바꾸는 코드
        if hold_person:
            for i in range(0, p_count):
                for j in range(0, len(hold_person)):
                    if(int(hold_person[j]) == Person_list[i].s_track_id):
                        if ( Person_list[i].pd_exist == 1) :                            #집었던 물건을 다시 내려놓음 (hold_person 리스트에서 지워줌) 나중에 
                            Person_list[i].action = "None"              
                            Person_list[i].s_color = (0,255,0)
                            hold_person.remove(hold_person[j]) 
                            break
                        elif ( Person_list[i].pd_exist == 0) :                          #집었던 물건을 숨김 (유지)
                            Person_list[i].action = "hide_product"
                            Person_list[i].s_color = (255,0,0)
                            #print("hide!")
                            break


        #물건을 가지거나 숨기고있을시 문과의 거리 계산
        for i in range(0, p_count) :
            if(str(Person_list[i].action) == "hold_product" or str(Person_list[i].action) == "hide_product"):    
                cv2.line(frame,(Person_list[i].s_center_x,Person_list[i].s_center_y),(door_centerx,door_centery),(255,255,0),5)
                #점사이의 거리계산 공식 : 루트((x1-x2)^2+(y1-y2)^2)
                ptod_distance = math.sqrt(pow(Person_list[i].s_center_x - door_centerx,2)+pow(Person_list[i].s_center_y - door_centery,2))
                #문과의 거리가 200 이하라면 LED on (나중에 바꿀예정)
                if(ptod_distance <= 60):
                    pin4.write(0)
                    pin5.write(1)
                    pin6.write(1)
                else:
                    pin4.write(1)
                    pin5.write(0)
                    pin6.write(0)


        
        #클래스에 할당된 값을 이용해 화면에 네모그리기        
        #door
        cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (255,255,255), 2)
        if(p_count > 0) :
            for i in range(0, p_count) :
                #person 그리기   
                print("x1: "+ str(Person_list[i].s_bx1) + " y1: " + str(Person_list[i].s_by1) + "x2: " + str(Person_list[i].s_bx2) +  " y2: " + str(Person_list[i].s_by2))
                cv2.rectangle(frame, (Person_list[i].s_bx1, Person_list[i].s_by1), (Person_list[i].s_bx2, Person_list[i].s_by2), Person_list[i].s_color, 2)
                cv2.rectangle(frame, (Person_list[i].s_bx1, Person_list[i].s_by1-30), (Person_list[i].s_bx1+(len(Person_list[i].s_class_name)+len(str(Person_list[i].s_track_id)))*17, Person_list[i].s_by1), Person_list[i].s_color, -1)          
                cv2.putText(frame, Person_list[i].s_class_name + "-" + str(Person_list[i].s_track_id) + " action : " + Person_list[i].action  ,((Person_list[i].s_bx1), Person_list[i].s_by1-10), 0, 0.75, (255,0,255),2)

        if(o_count > 0) :
            for i in range(0, o_count) : 
                if(Product_list[i].s_class_name != "knife"):
                    #product 그리기
                    cv2.rectangle(frame, (Product_list[i].s_bx1, Product_list[i].s_by1), (Product_list[i].s_bx2, Product_list[i].s_by2), Product_list[i].s_color, 2)
                    cv2.putText(frame, Product_list[i].s_class_name + "-" + str(Product_list[i].s_track_id) + " is_paid : " + str(Product_list[i].is_paid) ,((Product_list[i].s_bx1), Product_list[i].s_by1-10), 0, 0.75, (255,0,0),2)
            
        print("문 과의 거리 : "+str(ptod_distance))
        
        #for i in range (0, len(holded_pd)):
            #print ("hold_pd : ", holded_pd[i], " i :", i)


        #holded pd 에 존재하는 id 를 가진 물건이 age 가 어느정도 넘어가면 holded 에서 지워준다? 잡았다가 놓으면 holded_pd 의 age 0으로 초기화후 한 50 되면 holded_pd 에서 지워줌
        #print(o_count)
   
        del Person_list[:]
        del Product_list[:]


        #add code end ----------------------------------------------------------------------------------------------------------------------------------------------------
    
        # if enable info flag then print details about each track
        if FLAGS.info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #FPS 화면출력
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        #print("FPS: %.2f" % fps)

        
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:

        app.run(main)
    except SystemExit:
        pass
