import cv2
import math
import numpy as np

class LaneDetection():
    def __init__(self, cap):
        self.cap = cv2.VideoCapture(cap)

    def run(self):
        while (self.cap.isOpened()):
            result, self.frame = self.cap.read()

            if result:
                copy = np.copy(self.frame)
                self.process()
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process(self):
        self.resize()

    def resize(self, width=480, height=360):
        self.frame = cv2.resize(self.frame, (width, height))

    def gray(self):
        return cv2.cvtColor(np.asarray(self.frame), cv2.COLOR_RGB2GRAY)

    def gauss(self):
        return cv2.GaussianBlur(self.frame, (5, 5), 0)

    def canny(self):
        return cv2.Canny(self.frame, 50, 150)

    def region(self):
        height, width = self.frame.shape[:2]
        mask = np.zeros_like(self.frame)

        # Vid 2
        bottom_left = [width * 0.05, height * 0.95]
        top_left = [width * 0.4, height * 0.4]
        bottom_right = [width * 0.9, height * 0.95]
        top_right = [width * 0.6, height * 0.4]

        roi_vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        mask_color = 255
        cv2.fillPoly(mask, roi_vertices, mask_color)
        masked_edges = cv2.bitwise_and(self.frame, mask)

        return masked_edges

    def make_points(self, average):
        slope, y_int = average
        y1 = self.frame.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)

        return np.array([x1, y1, x2, y2])

    def average(self, lines):
        left = []
        right = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            
            slope = parameters[0]
            y_int = parameters[1]

            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        
        left_line = self.make_points(self.frame, left_avg)
        right_line = self.make_points(self.frame, right_avg)

        return np.array([left_line, right_line])

    def display_lines(self, lines):
        lines_image = np.zeros_like(self.frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 15)
        return lines_image

test = LaneDetection('./vid_2.mp4')
test.run()