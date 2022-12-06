import cv2
import numpy

class WriteCardInfo:
    def __write_info__(frame, x1, y1, bgr):
        cv2.putText(frame, "Lorem ipsum dolor sit", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        cv2.putText(frame, "Lorem ipsum dolor sit", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        cv2.putText(frame, "Lorem ipsum dolor sit", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

