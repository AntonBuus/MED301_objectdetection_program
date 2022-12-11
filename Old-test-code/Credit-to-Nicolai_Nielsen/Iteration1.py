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

    def draw_label(self, img, text, pos, bg_farve):
        self.font_face = cv2.FONT_HERSHEY_TRIPLEX
        self.scale = 1
        self.farve = bg_farve
        self.thickness = 3
        self.margin = 20
        self.txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)

        end_x = pos[0] + self.txt_size[0][0] + self.margin
        end_y = pos[1] - self.txt_size[0][1] - self.margin

        cv2.putText(img, text, pos, self.font_face, self.scale, self.farve, 1, cv2.LINE_AA)

    def wrappedtext(self, img, text, bg_farve):

            wrapped_text = textwrap.wrap(text, width=80)
            self.scale = 1
            self.font_face = cv2.FONT_HERSHEY_TRIPLEX
            self.thickness = 2
            self.farve = (255,255,255)
            self.bg_farve = bg_farve

            x, y = 10, 40

            for i, line in enumerate(wrapped_text):
                self.textsize = cv2.getTextSize(line, self.font_face, self.scale, self.thickness)[0]
                gap = self.textsize[1] + 10
                y = int((img.shape[0] + self.textsize[1]) / 2) + i * gap
                x = int((img.shape[1] - self.textsize[0]) / 2)


                cv2.putText(img, line, (x,y-400), self.font_face, self.scale, self.farve, self.thickness, cv2.LINE_AA)

    
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
                bgr = (182, 0, 255)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)                
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
        
            self.correct = "Det er rigtigt!"
            self.wrong = "Det er forkert :("
            self.scan_card = "Placer to kort i rammen, for at begynde"
            self.color = 255,255,255
            self.pos =  550,950
            self.correct_color = (57, 255, 8)
            self.wrong_color = 0, 0, 255


            self.text1 = """The monument here is the original model in plaster from 1824 to 25: Pius 7 was the pope from 1800 until his death in 1823. The marble monument was revealed in St. Peter`s Basilica in 1831. It is the sign of Thorvaldsens fame that he received the order on Pope Pius 7. tomb of the catholicism`s main church, even though he was protestant. The aging pope sits on his throne with the tiara on his head. With the stretched hand he makes a kind of gesture while looking forward to us. The allegorical figure of women to the right, The Divine Strength, on the other hand, looks up against the sky and the woman on the left, The Heavenly Wisdom, looks down, thoughtfully immersed in the book, the Bible she holds. The Pope in the middle thus becomes the balanced mediator of a message consisting of equal parts of Christian sentiment andz Christian wisdom."""

            self.text2 = "This statue was made in the period 1827-1828: The kneeling angel with a flower wreath on his head and a clamshell in his hands created Thorvaldsen as a baptismal font for Our Lady Church in the 1820s."

            self.text3 = "This statue was made in 1833: The band that the geniuses hold is a so-called banderole either with the notes or the lyrics to what they sing. The wave movement of the banner across the relief can be seen as an image of the vocal cords or, rather, of the falling and rising tones that voices produce during singing."

            screen_size = 1920,1080

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (screen_size)) #god skærmstørrelse
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow('YoloV5, Regular Screen', frame)
            cv2.rectangle(frame, (0,0),(screen_size), (0,0,0),-1)
            cv2.rectangle(frame, (0,0),(screen_size), (2255,255,255),75)
            start_1 = cv2.rectangle(frame, (670, 540), (900, 900), (255,255,255), 2)
            start_2 = cv2.rectangle(frame, (1020, 540), (1250, 900), (255,255,255), 2)


            frame = self.plot_boxes(results, frame)
            
            length = len(self.detections)
            
            if length == 0:
                self.draw_label(frame, self.scan_card, (self.pos), (self.color))
            
            elif length == 1:
                self.draw_label(frame, "Placer et kort mere", (self.pos), (self.color))
       

            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[0]):
                self.draw_label(frame, self.correct, (self.pos), (57, 255, 8))
                self.wrappedtext(frame, self.text1, (255,255,225))
                start_1 = cv2.rectangle(frame, (670, 540), (900, 900), (57, 255, 8), 2)
                start_2 = cv2.rectangle(frame, (1020, 540), (1250, 900), (57, 255, 8), 2)

            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[1]):
                self.draw_label(frame, self.correct, (self.pos), (57, 255, 8))
                self.wrappedtext(frame, self.text3, (self.color))
                start_1 = cv2.rectangle(frame, (670, 540), (900, 900), (57, 255, 8), 2)
                start_2 = cv2.rectangle(frame, (1020, 540), (1250, 900), (57, 255, 8), 2)
            
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[2]):
                self.draw_label(frame, self.correct, (self.pos), (57, 255, 8))
                self.wrappedtext(frame, self.text2, (self.color))
                start_1 = cv2.rectangle(frame, (670, 540), (900, 900), (57, 255, 8), 2)
                start_2 = cv2.rectangle(frame, (1020, 540), (1250, 900), (57, 255, 8), 2)
                
            elif self.detections != self.correct_answers:
                self.draw_label(frame, self.wrong, (self.pos), (0,0,255))
                start_1 = cv2.rectangle(frame, (670, 540), (900, 900), (0, 0, 255), 2)
                start_2 = cv2.rectangle(frame, (1020, 540), (1250, 900), (0, 0, 255), 2)
            
            
            cv2.imshow('YoloV5 Detection - Dark Screen', frame)
        
 
            if cv2.waitKey(1) & 0xFF == ord('p'):
                 print(self.detections) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
      
        cap.release()
        
        
# Create a new object and execute.
detector = CardDetection(capture_index=1, model_name='Supergood2.pt')
detector()