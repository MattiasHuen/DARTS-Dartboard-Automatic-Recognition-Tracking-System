import cv2
import numpy as np
import time

from src.config import *
from src.calibration import calibrate_from_video
from src.visualization import draw_fields
from src.field_detection import detect_fields

class DartBoard:
    """Detect and manage a dartboard."""

    def __init__(self):
        # Frames
        self.frame = None
        self.frame_gray = None

        # Calibration results
        self.calibrated = False
        self.board_homography = None
        self.bullseye = None
        self.outer_ellipse = None
        self.inner_ellipse = None

        #Scoring Results
        self.fields_detected = False
        self.rings = None
        self.sectors = None
        self.fields = None
        self.field_map = None

    def update_frame(self, frame):
        """ Update frame"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid Frame Provided")

        self.frame = frame
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def calibrate(self, frame):
        """ Calibrate from frame """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid Frame Provided")

        if not self.calibrated:
            result =  calibrate_from_video(frame)
            self.board_homography = result
            self.bullseye  = result["bullseye"]
            self.outer_ellipse = result["outer"]
            self.inner_ellipse = result["inner"]
            self.calibrated = True
        return True

    def detect_scoring_fields(self):
        if self.calibrated is False:
            raise ValueError("Board must be calibrated first")

        if not self.fields_detected:
            result = detect_fields(self.frame, self.board_homography)

            self.rings = result["rings"]
            self.sectors = result["sectors"]
            self.fields = result["fields"]
            self.field_map = result["field_map"]

            self.fields_detected = True

        return True

    
    def draw_fields(self):
        if self.fields_detected is False:
            raise ValueError("Fields must be found first")
        
        return draw_fields(self.frame, self.fields, self.field_map)

    def inner_drawn_fields(self, with_fields: bool = True):
        if self.fields_detected is False:
            raise ValueError("Fields must be found first")

        output = self.draw_fields()

        mask = (self.field_map >= 0).astype(np.uint8) * 255
        
        isolated = cv2.bitwise_and(output, output, mask=mask)

        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        if with_fields:
            return isolated[y:y+h, x:x+w]
        else:
            return self.frame[y:y+h, x:x+w]
        
    

        

   