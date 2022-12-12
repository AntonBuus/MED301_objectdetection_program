import torch
import numpy as np
import cv2
from time import time
from PIL import Image
import collections
import textwrap 


class CardDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.correct_answers = [["ya", "sa"], ["li", "ta"],["ma", "n"]]

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture(self.capture_index)


    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def load_image (self, frame, img, position):
        self.overlay = cv2.imread(img)

        # Get Image dimensions
        self.overlay_height, self.overlay_width, _ = self.overlay.shape

        # Decide X,Y location of overlay image inside video frame. 
        x, y = position

        frame[ y:y+self.overlay_height , x:x+self.overlay_width ] = self.overlay


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        self.detections = []
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.88:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                self.detections.append(self.class_to_label(labels[i]))

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True: 
            #Colors
            self.black = 0,0,0
            self.white = 255,255,255
            self.correct_color = 57,255,8
            self.yellow = 0,230,255
            self.wrong_color = 0,0,255
            
            #Image loading and position
            self.background = cv2.imread('Thorvaldsens.png')
            self.lower_position = 510, 942
            self.text_position = 301, 142
            
            self.place_start = "start.png"
            self.place_start2 = "One_more.png"
            self.correct_card = "Correct.png"
            self.wrong_card = "wrong.png"
            self.text1 = "Text1.png"
            self.text2 = "Text2.png"
            self.text3 = "text3.png"
            self.tv = "tvbillede.png"
            
            #Rectangles for placement of card
           #self.start_pos1 = (670, 540) # Starting Point for Rectangle 1 
            #self.end_pos1 = (900, 900) #Ending Point for Rectangle 1
            self.start_pos1 = (640, 510) # Starting Point for Rectangle 1 
            self.end_pos1 = (900, 900) #Ending Point for Rectangle 1            

            
            #self.start_pos2 = (1020, 540) # Starting Point for Rectangle 2 
            #self.end_pos2 = (1250, 900) #Ending Point for Rectangle 2
            self.start_pos2 = (990, 510) # Starting Point for Rectangle 2 
            self.end_pos2 = (1250, 900) #Ending Point for Rectangle 2
            
            #Screen Size
            screen_size = 1920,1080


            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (screen_size)) #god skærmstørrelse
            

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow('YoloV5, Regular Screen', frame)
            
            #Background
            frame = self.background
            #cv2.rectangle(frame, (0,0),(screen_size), (0,0,0),-1) # Black Screen Background
            
            #Rectangles around cards 
            start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.black), 2)
            start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.black), 2)


            frame = self.plot_boxes(results, frame)
            # Checking the length of the detections lists in plot_boxes function. 
            length = len(self.detections)
            
            # Conditional Statements, checking the current 'labels' detected compared to the correct answers list. 
            if length == 0:
                self.load_image (frame, self.tv, self.text_position)
                self.load_image (frame, self.place_start, self.lower_position)
                #self.draw_label(frame, self.scan_card, (self.pos), (self.black))
            
            elif length == 1:
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.yellow), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.yellow), 2)
                self.load_image (frame, self.place_start2, self.lower_position)

            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[0]):
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)
                self.load_image (frame, self.correct_card, self.lower_position)
                self.load_image (frame, self.text1, self.text_position)

            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[1]):
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)
                self.load_image (frame, self.correct_card, self.lower_position)
                self.load_image (frame, self.text2, self.text_position)
            
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[2]):
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)
                self.load_image (frame, self.correct_card, self.lower_position)
                self.load_image (frame, self.text3, self.text_position)
                
            elif self.detections != self.correct_answers:
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.wrong_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.wrong_color), 2)
                self.load_image (frame, self.wrong_card, self.lower_position)
            
            
            cv2.imshow('YoloV5 Detection - UI', frame)
        
 
            if cv2.waitKey(1) & 0xFF == ord('p'):
                 print(self.detections) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
      
        cap.release()
        
        
# Create a new object and execute.
detector = CardDetection(capture_index=0, model_name='Supergood2.pt')
detector()