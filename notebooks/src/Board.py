import cv2
import numpy as np
import time

from src.config import *
from src.calibration import calibrate_from_video
from src.visualization import draw_fields
from src.field_detection import detect_fields
from src.dart_detection import find_dart, ellipse_mask
from src.scoring import field_at

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


    def detect_dart(self, before, tip_extension=5):
        """Locate a newly landed blue dart and score its estimated tip.

        `before` must be a FULL-SIZE frame from before this throw.
        Current frame is `self.frame`, set using update_frame().
        """
        if not self.fields_detected:
            raise ValueError("Detect the scoring fields first")

        # Allow shaft and white flights to extend slightly past the board boundary.
        center, (width, height), angle = self.outer_ellipse
        padded_ellipse = (center, (width * 1.25, height * 1.25), angle)
        mask = ellipse_mask(self.frame.shape, padded_ellipse)

        result = find_dart(before, self.frame, mask, tip_extension=tip_extension)
        result["field"] = None
        result["score"] = None

        if result["found"]:
            x, y = np.rint(result["tip"]).astype(int)
            height, width = self.field_map.shape
            if 0 <= x < width and 0 <= y < height:
                result["field"], result["score"] = field_at(
                    x, y, self.fields, self.field_map
                )
            else:
                result["field"], result["score"] = "Miss", 0

        return result

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

    def outer_drawn_fields(self, with_fields: bool = True):
        if self.fields_detected is False:
            raise ValueError("Fields must be found first")

        output = self.draw_fields() if with_fields else self.frame.copy()

        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)

        (cx, cy), (width, height), angle = self.outer_ellipse
        cv2.ellipse(mask,
        (int(cx), int(cy)),
        (int(width / 2), int(height / 2)),
        angle,
        0,360,255,-1
        )
        
        isolated = cv2.bitwise_and(output, output, mask=mask)

        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        return isolated[y:y+h, x:x+w], (x, y)
        
    

        

   