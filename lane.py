import cv2
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt

filename = "./challenge.mp4"
file_size = (1280, 720)
scale_ratio = 1

output_filename = "vid_2_output.mp4"
output_frames_per_second = 20.0

class Lane:
    def __init__(self, orig_frame):
        self.orig_frame = orig_frame
        self.lane_line_markings = None

        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        self.roi_points = np.float32([
            (int(0.456*width), int(0.544*height)),
            (0, height-1),
            (int(0.958*width), height-1),
            (int(0.6183*width), int(0.544*height))
        ])

        self.padding = int(0.25 * width)
        self.desired_roi_points = np.float32([
            [self.padding, 0],
            [self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0] - self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0] - self.padding, 0]
        ])

        self.histogram = None

    def calculate_car_position(self):
        pass

    def calculate_curvature(self):
        pass

    def calculate_histogram(self, frame=None):
        if frame is None:
            frame = self.warped_frame

        self.histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)
        return self.histogram

    def display_curvature_offset(self):
        pass

    def get_lane_line_previous_window(self):
        pass

    def get_lane_line_indices_sliding_windows(self):
        pass

    def get_line_markings(self, frame=None):
        if frame is None:
            frame = self.orig_frame

        # Convert to HLS (Hue, Lightness, Saturation)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Perform Sobel edge detection on lightness channel
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        # Perform binary thresholding on saturation channel
        s_channel = hls[:, :, 2]
        _, s_binary = edge.threshold(s_channel, (130, 255))

        # Perform binary thresholding on red channel
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))

        # Binary 'and' to reduce noise
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        # Combine possible edges with binary 'or'
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        return self.lane_line_markings

    def get_line_markings_canny(self, frame=None):
        if frame is None:
            frame = self.orig_frame

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.Canny(frame, 50, 150)

        self.lane_line_markings = frame
        return self.lane_line_markings

    def histogram_peak(self):
        pass

    def overlay_lane_lines(self):
        pass

    def perspective_transform(self, frame=None):
        if frame is None:
            frame = self.lane_line_markings
        
        # Calculate transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)

        # Calculate inverse of the transformation matrix
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)

        # Perform transformation
        self.warped_frame = cv2.warpPerspective(frame, self.transformation_matrix, self.orig_image_size, flags=(
            cv2.INTER_LINEAR
        ))

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        pass

def main():
    # Load Video
    cap = cv2.VideoCapture(filename)

    # Write to Video Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,  
                           fourcc, 
                           output_frames_per_second, 
                           file_size)

    # Process Video
    while (cap.isOpened()):
        success, frame = cap.read()
        
        if success:
            # Resize frame
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Store original
            original_frame = frame.copy()
            lane_obj = Lane(orig_frame=original_frame)

            # Perform thresholding to isolate lane lines
            lane_line_markings = lane_obj.get_line_markings_canny()

            # Plot region of interest
            # lane_obj.plot_roi(frame=None, plot=False)

            # Transform to bird's eye view
            warped_frame = lane_obj.perspective_transform()

            histogram = lane_obj.calculate_histogram()	
            
            # left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
            #     plot=False)

            # lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
            
            # frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

            # lane_obj.calculate_curvature(print_to_terminal=False)

            # lane_obj.calculate_car_position(print_to_terminal=False)
            
            # frame_with_lane_lines2 = lane_obj.display_curvature_offset(
            #     frame=frame_with_lane_lines, plot=False)
                        
            # result.write(frame_with_lane_lines2)
                    
            cv2.imshow("Frame", warped_frame) 	

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break  
        else:
            break
                        
    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()