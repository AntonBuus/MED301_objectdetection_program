import os
import cv2
import torch
import numpy as np
import collections


os.environ['KMP_DUPLICATE_LIB_OK']='True'

 
#Loading current best model
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Supergood.pt', force_reload=True)

#Empty List, where detected cards can be added
detections = []

#Correct Scenarios
rigtig = ["C1", "C2"]
rigtig2 = ["C3", "C4"]
rigtig3 = ["C5", "C6"]


#Function creating labels, that can be dynamically adjusted
def __draw_label(img, text, pos, bg_farve):
    font_face = cv2.FONT_HERSHEY_TRIPLEX
    scale = 1
    farve = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 20
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_farve, thickness)
    cv2.putText(img, text, pos, font_face, scale, farve, 1, cv2.LINE_AA)


#________ Variables used for draw_label 
test = "Scan Card"

farve = 255,0,0

pos =  150,150
#________ Variables used for draw_label 

# Setting video Capture device, which by default is the webcam. 
cap = cv2.VideoCapture(0)

#While loop for when the camera is activated. This is a loop that is continued throughout its running time. 
while cap.isOpened():
    
    # Starts the loop, where ret acts a boolean.
    ret, frame = cap.read()
    #Setting our trained model, to a frame.
    results = model(frame)
    # im predictions (tensor)
    results.xyxy[0]  
    # Using Numpy to render our results. Squeeze function removed single-dimensional enteries from the shape of an array
    frame = np.squeeze(results.render())
    
    #___________- Masking pixels from stackoverflow https://stackoverflow.com/questions/54644357/how-to-detect-black-colour-in-a-video-cam-using-opencv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_red = np.array([110,50,50]) 
    upper_red = np.array([130,255,255]) 
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
     #___________- Masking pixels
    
    #Ret as mentioned is a boolean value. it checks for a frame. 
    if ret == True:
        # draw the label into the frame
        __draw_label(frame, test, (pos), (farve))

        # Display the resulting frame
        cv2.imshow('Frame', mask)
    
    
    # This is for detecting the cards, and adding them to the detection array, which will be sent over to our algorithm.
    if cv2.waitKey(1) & 0xFF == ord('d'):
        detectedCard = results.pandas().xyxy[0]['name']
        cardName = detectedCard.to_string()[5:]
      
        cardConvert = {"6S": 'C1', "10C": 'C2', 
                        "7H": 'C3', "QD": 'C4',
                        "3D": 'C5', "5C": 'C6'}


        card = cardConvert.get(cardName, "Card not recognized, please point camera at the corner of the card.")

        if len(card) > 3:
            print(card)
            
        else:
            detections.append(card)
            print("Succesfully entered a card")
            print(detections)
            text = cv2.putText(frame, "Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        if collections.Counter(detections) == collections.Counter(rigtig) or collections.Counter(rigtig2) or collections.Counter(rigtig3):
            farve = 49,140,0
            test = "Correct!"
            __draw_label(frame, test, (pos), (farve))
            print ("The lists l1 and l2 are the same") 
            print(test)
        else: 
            farve = 0,8,247
            test = "Not Correct"
            __draw_label(frame, test, (pos), (farve))
            print ("The lists l1 and l2 are not the same") 
            
            
    # explain
    if cv2.waitKey(1) & 0xFF == ord('s'):
        detectionsString = ",".join(detections) + '\n'

        print(detectionsString)
    
    # Prints whatever is in the detection array, can be used to check if any of the inputs are incorrect.
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(detections)

    # Prints whatever is in the detection array, can be used to check if any of the inputs are incorrect.
    if cv2.waitKey(1) & 0xFF == ord('c'):
        detections.clear()
        farve = 255,0,0
        test = "Scan Card"
        __draw_label(frame, test, (pos), (farve))
        print("Succesfully cleared detection array")
        print(detections)

    # Prints whatever is in the detection array, can be used to check if any of the inputs are incorrect.
    if cv2.waitKey(1) & 0xFF == ord('l'):
        lastcard = len(detections)-1
        detections.pop(lastcard)
        print("Succesfully cleared detełtion array")
        print(detections)
        
    # This is for manually inputting the cards, if the detection for some reason fails.
    if cv2.waitKey(1) & 0xFF == ord('m'):
        manual_Detection = input("Enter card suit and rank: ") # The format is f.x King of hearts = KH
        detections.append(manual_Detection)
        print("Succesfully entered a card")
        print(detections)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()