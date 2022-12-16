import torch
import cv2
import collections

 
class Objectdetection:
    """
    Class containing functions for object detection
    """
    def __init__(self, video_index, model_file):
        """
        This function is a constructor for the class. It initiates the class based on the values that being inserted into the constructor.
        video_index, refers to the camera the user wants to use.
        model_file, is the model that user wants to use. This has been provided by us. 
        This function additionally checks if a GPU is available. If it isn't the CPU will be used
        There are also three lists, one containing the combinations resulting in a correct answer, 
        and one list each for the two categories of cards - statue cards and question cards
        """
        self.video_index = video_index
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_file, force_reload=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.correct_answers = [["di", "ba"], ["li", "g"],["ya", "n"], ["ma","wa"]]
        self.only_question = [["di"], ["li"],["ya"], ["ma"]]
        self.only_statue = [["ba"], ["g"],["n"], ["wa"], ["sa"], ["ka"] ,["sa"],["ka"]]

    def live_feed(self):
        """
        This functions return a cv2 videocapturing object, using the index set by the user (self.video_index)
        """
        return cv2.VideoCapture(self.video_index)

    
    def load_image (self, frame, img, position):
        """
        This function takes a frame, image and position as input. 
        The purpose of this function is to 'update' the UI (the images displayed), based on the cards being detected (the users response)
        Position refers to the position on the frame, where the image should be displayed. 
        """
        self.overlay = cv2.imread(img)

        # Image dimensions. Used to ensure the image can fit in the frame
        self.overlay_height, self.overlay_width, _ = self.overlay.shape

        # Decide X,Y location of overlay image inside video frame. 
        x, y = position

        #Sets the frame to the image. Addtionally ensures that image can fit in the frame.
        frame[ y:y+self.overlay_height , x:x+self.overlay_width ] = self.overlay


    def detections_in_frame(self, frame):
        """
       Takes a frame as input and return the labels and coordinates detected by the model.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def names_to_list(self, results, frame):
        """
        Takes a frame as input. 
        The results are being provided by the detections_in_frame function. It has the labels and coordinates picked up by the model.
        Additionally this allows us to set a threshold for the coordinates - i.e setting a number for the confidence level from which objects should be detected
        The purpose of this function is to fill the the empty list, with the labels of the objects being detected by the model. 
        This is done so it can be checked if the user has answered correctly or not, and to create prompts for the user. 
        This function can also be used to add bounding boxes and display the labels on the frame. However, this is not needed
        """
        self.detections = []
        labels, cord = results
        n = len(labels)
        for i in range(n):
            obj_pos = cord[i]
            if obj_pos[4] >= 0.88:
                names = self.classes[int(labels[i])]
                self.detections.append(names)

        return frame

    def __call__(self):
        """
        This function is called upon, when an instance of the class is created. 
        The function contains a while loop that read through each frame from the live feed
        """
        
        cap = self.live_feed()
        assert cap.isOpened()
      
        while True: 
            #Colors
            self.black = 0,0,0
            self.white = 255,255,255
            self.correct_color = 57,255,8
            self.yellow = 0,230,255
            self.wrong_color = 0,0,255
            
            #Image loading and position
            self.background = cv2.imread('PNGs/Thorvaldsens.png')
            self.label_position = 510, 50
            self.frame_position = 21, 42
            
            self.start = "PNGs/Start.png"
            self.om_question = "PNGs/om_question.png"
            self.om_statue = "PNGs/om_statue.png"
            self.wrong = "PNGs/wrong_combo.png"

            self.christ = "PNGs/Christ.png"
            self.jason = "PNGs/Jason.png"
            self.nico = "PNGs/Nico.png"
            self.paven = "PNGs/Paven.png"
            
            #Rectangles for placement of card
            self.start_pos1 = (678, 590) # Starting Point for Rectangle 1 
            self.end_pos1 = (900, 970) #Ending Point for Rectangle 1            

            self.start_pos2 = (1000, 590) # Starting Point for Rectangle 2 
            self.end_pos2 = (1225, 970) #Ending Point for Rectangle 2
            
            #Screen Size
            screen_size = 1920,1080


            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (screen_size)) #god skærmstørrelse
            

            results = self.detections_in_frame(frame)
            frame = self.names_to_list(results, frame)
            cv2.imshow('YoloV5, Regular Screen', frame)
            
            #Background
            frame = self.background
            
            #Rectangles around cards 
            start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.black), 2)
            start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.black), 2)


            frame = self.names_to_list(results, frame)
            # Checking the length of the detections lists in names_to_list function. 
            length = len(self.detections)
            
            # Conditional Statements, checking the current 'labels' detected compared to the correct answers list. 
            if length == 0:
                self.load_image (frame, self.start, self.label_position)

            
            #This following to elif conditions checks what card has been placed and accordingly displays the right frame.
            elif collections.Counter(self.detections) == collections.Counter(self.only_statue[0]) or collections.Counter(self.detections) == collections.Counter(self.only_statue[1]) or collections.Counter(self.detections) == collections.Counter(self.only_statue[2]) or collections.Counter(self.detections) == collections.Counter(self.only_statue[3]) or collections.Counter(self.detections) == collections.Counter(self.only_statue[4]) or collections.Counter(self.detections) == collections.Counter(self.only_statue[5]) :
                self.load_image (frame, self.om_question, self.label_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.yellow), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.yellow), 2)
                
            elif collections.Counter(self.detections) == collections.Counter(self.only_question[0]) or collections.Counter(self.detections) == collections.Counter(self.only_question[1]) or collections.Counter(self.detections) == collections.Counter(self.only_question[2]) or collections.Counter(self.detections) == collections.Counter(self.only_question[3]):
                self.load_image (frame, self.om_statue, self.label_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.yellow), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.yellow), 2)

            #The following four elif statements, check if a pair within the correct answer list has been placed and detected
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[0]):
                self.load_image (frame, self.paven, self.frame_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)


            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[1]):
                self.load_image (frame, self.jason, self.frame_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)

            
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[2]):
                self.load_image (frame, self.christ, self.frame_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)

                
                
            elif collections.Counter(self.detections) == collections.Counter(self.correct_answers[3]):
                self.load_image (frame, self.nico, self.frame_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.correct_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.correct_color), 2)

                               
             #Checks if the cards are wrongly matched   
            elif self.detections != self.correct_answers:
                self.load_image (frame, self.wrong, self.label_position)
                start_1 = cv2.rectangle(frame, self.start_pos1, self.end_pos1, (self.wrong_color), 2)
                start_2 = cv2.rectangle(frame, self.start_pos2, self.end_pos2, (self.wrong_color), 2)

                   
            cv2.imshow('YoloV5 Detection - UI', frame)
        
 
            if cv2.waitKey(1) & 0xFF == ord('p'):
                 print(self.detections) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
      
        cap.release()
        
        
# Creates a new instance of class
obj_1 = Objectdetection(video_index=1, model_file='Supergood2.pt')
obj_1()