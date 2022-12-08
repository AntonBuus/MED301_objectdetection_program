import torch
import numpy as np
import cv2
from time import time
from PIL import Image

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

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        global kamera 
        kamera = cv2.VideoCapture(self.capture_index)
        return kamera

   #def rotate_camera(self):
       # global r_cam
        #r_cam = cv2.rotate(kamera, cv2.cv2.ROTATE_90_CLOCKWISE)
        #return r_cam

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

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.83:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (255, 255, 0)
                
                
                # img = frame
                
                # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #unchanged box
                # cv2.rectangle(frame, (x1-100, y1-100), (x2+200, y2+200), bgr, 2)

                
                # cv2.line(frame, (x1+250, y1+100), (x2+400, y2+100), bgr, 2)
                if self.class_to_label(labels[i]) == "ya":
                    cv2.rectangle(frame, (x1+400, y1-150), (x2+600, y2+100), bgr, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #unchanged box
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1+415, y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

                    cv2.rectangle(frame, (x1-400, y1+100), (x2-200, y2), bgr, 2)
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                    cv2.putText(frame, "Lorem ipsum dolor sit", (x1-375, y1+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                    
                
                
                # cv2.putText(frame, """Lorem ipsum dolor sit amet\n

                # Lorem ipsum dolor sit amet \n

                # Lorem ipsum dolor sit amet""", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                # write_info(frame, x1, y1, bgr)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

                # cv2.line(frame, (0, 0), (0, 200), bgr, 2)
                # cv2.imshow('image',img)
                # cv2.imshow(1)

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
        
            ret, frame = cap.read()
            assert ret
            
            # frame = cv2.resize(frame, (416,416))
            # frame = cv2.resize(frame, (windowsize, windowsize))
            #frame = cv2.resize(frame, (1200,900)) #god skærmstørrelse (prøver noget andet nedenunder - Gabe)
            frame = cv2.resize(frame, (1920,1080)) #god skærmstørrelse
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow('YoloV5, Regular Screen', frame)
            cv2.rectangle(frame, (0,0),(1920,1080), (0,0,0),-1)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 10/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YoloV5 Detection - Dark Screen', frame)

            cv2.setWindowProperty("YoloV5", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


 
            if cv2.waitKey(5) & 0xFF == 27:
                break
      
        cap.release()
        
        
# Create a new object and execute.
detector = CardDetection(capture_index=0, model_name='Supergood2.pt')
detector()