import torch
import numpy as np
import cv2
from time import time
from PIL import Image
import collections
import textwrap 


# img =cv2.imread('GoAB.jpg',0)
# img1 = Image.open(r"C:\GitHub\CardDetect\GoAB.jpg")

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
        self.farve = (0, 0, 0)
        self.thickness = cv2.FILLED
        self.margin = 20
        self.txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)

        end_x = pos[0] + self.txt_size[0][0] + self.margin
        end_y = pos[1] - self.txt_size[0][1] - self.margin

        cv2.rectangle(img, pos, (end_x, end_y), bg_farve, self.thickness)
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

                #cv2.rectangle(img, pos, (end_x, end_y), bg_farve, self.thickness, -1)

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
            if row[4] >= 0.83:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (182, 0, 255)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.rectangle(frame, (x1-50, y1-250), (x2+100, y2+120), bgr, 2)
                #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)                
                self.detections.append(self.class_to_label(labels[i]))
                
                
                # img = frame
                
                # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #unchanged box
                # cv2.rectangle(frame, (x1-100, y1-100), (x2+200, y2+200), bgr, 2)
                # cv2.line(frame, (x1+250, y1+100), (x2+400, y2+100), bgr, 2)
                #if self.class_to_label(labels[i]) == "ya":
                #    cv2.rectangle(frame, (x1+400, y1-150), (x2+600, y2+100), bgr, 2)
                #    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #unchanged box
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                #    cv2.rectangle(frame, (x1-400, y1+100), (x2-200, y2), bgr, 2)
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                #    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                # cv2.putText(frame, """Lorem ipsum dolor sit amet\n

                # Lorem ipsum dolor sit amet \n

                # Lorem ipsum dolor sit amet""", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                # write_info(frame, x1, y1, bgr)
                #cv2.putText(frame, self.class_to_label(labels[i]), (self.x1, self.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.bgr, 2)

                # cv2.line(frame, (0, 0), (0, 200), bgr, 2)
                # cv2.imshow('image',img)
                # cv2.imshow(1)
                
                #return selfx1, y1, x2, y2

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
            self.color = 255,250,150
            self.pos =  50,50

            self.text1 = "Pius 7 was the pope from 1800 until his death in 1823. The monument here is the original model in plaster from 1824 to 25, the marble monument was revealed in St. Peter’s Basilica in 1831. It is the sign of Thorvaldsen’s fame that he received the order on Pope Pius 7. tomb of the catholicism’s main church, even though he was protestant. The aging pope sits on his throne with the tiara on his head. With the stretched hand he makes a kind of gesture while looking forward to us. The allegorical figure of women to the right, The Divine Strength (A144), on the other hand, looks up against the sky and the woman on the left, The Heavenly Wisdom (A143), looks down, thoughtfully immersed in the book, the Bible she holds. The Pope in the middle thus becomes the balanced mediator of a message consisting of equal parts of Christian sentiment andz Christian wisdom."

            self.text2 = "The kneeling angel with a flower wreath on his head and a clamshell in his hands created Thorvaldsen as a baptismal font for Our Lady Church in the 1820s."
            self.text3 = "The band that the geniuses hold is a so-called banderole either with the notes or the lyrics to what they sing. The wave movement of the banner across the relief can be seen as an image of the vocal cords or, rather, of the falling and rising tones that voices produce during singing."

            #text_1 = "Lorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsum"
            #wrapped_text = textwrap.wrap(text_1, width=35)

            screen_size = 1920,1080

            ret, frame = cap.read()
            assert ret

            #end_time = time()
            #fps = 10/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
            # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            frame = cv2.resize(frame, (screen_size)) #god skærmstørrelse
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow('YoloV5, Regular Screen', frame)
            cv2.rectangle(frame, (0,0),(screen_size), (0,0,0),-1)


            frame = self.plot_boxes(results, frame)
            
            length = len(self.detections)
            
            if length == 0:
                self.draw_label(frame, self.scan_card, (self.pos), (self.color))
            
            elif length == 1:
                self.draw_label(frame, "Placer et kort mere", (self.pos), (self.color))
            
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[0]):
                self.draw_label(frame, self.correct, (self.pos), (49,140,0))
                self.wrappedtext(frame, self.text1, (0,255,221))

            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[1]):
                self.draw_label(frame, self.correct, (self.pos), (49,140,0))
                self.wrappedtext(frame, self.text3, (0,255,221))
            
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[2]):
                self.draw_label(frame, self.correct, (self.pos), (49,140,0))
                self.wrappedtext(frame, self.text2, (0,255,221))
                
            elif self.detections != self.correct_answers:
                self.draw_label(frame, self.wrong, (self.pos), (0,8,247))
            
            
            cv2.imshow('YoloV5 Detection - Dark Screen', frame)
        
 
            if cv2.waitKey(1) & 0xFF == ord('p'):
                 print(self.detections) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
      
        cap.release()
        
        
# Create a new object and execute.
detector = CardDetection(capture_index=0, model_name='Supergood2.pt')
detector()