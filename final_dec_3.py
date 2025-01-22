from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2, Preview

# Load a pretrained YOLOv8n model
model = YOLO('yolov8s.pt')

# Initialize the PiCamera2 and configure settings
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})  # Set resolution to 640x480
picam2.configure(config)
picam2.start()

# Define the codec and create VideoWriter object to save output (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'results.mp4'
fps = 30  # Set FPS (you can adjust this based on performance)
out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

def lane_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Color filtering: convert to HLS and filter yellow and white
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([10, 0, 100]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_frame = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Blur and Edge Detection
    blur = cv2.GaussianBlur(masked_frame, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Define Region of Interest (ROI) to exclude unwanted areas
    height, width = edges.shape
    side_lanes_roi = np.array([[(0, height), (width * 0.4, height * 0.6), (width * 0.6, height * 0.6), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, side_lanes_roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Line Detection with refined parameters
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)
    
    # Draw detected lanes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green color for lane lines
    return frame


def calculate_distance(box1, box2):
    # Calculate the center points of the bounding boxes
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)

    # Calculate the distance between the centers
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Convert pixel distance to real-world units (e.g., meters)
    real_world_distance = distance / 30.0  # Adjust scale as needed
    return real_world_distance

while True:
    # Capture a frame from the PiCamera2
    frame = picam2.capture_array()

    # Run inference on the current frame
    results = model(frame)

    # Store vehicle bounding boxes
    vehicle_boxes = []

    # Visualize results and draw lane markings
    for r in results:
        im_bgr = r.plot()
        for box in r.boxes.xyxy:  # Assuming boxes are in xyxy format
            vehicle_boxes.append(box.tolist())  # Convert to list

        # Detect lanes and draw on the frame
        im_bgr = lane_detection(im_bgr)

    # Check for distances only for the front vehicle
    front_vehicle_distance = None
    if len(vehicle_boxes) > 1:
        # Assume the first vehicle in the list is the one closest to the camera
        front_vehicle = vehicle_boxes[0]
        closest_distance = float('inf')

        # Compare with all other vehicles to find the closest one in front
        for box in vehicle_boxes[1:]:
            distance = calculate_distance(front_vehicle, box)
            if distance < closest_distance:
                closest_distance = distance
                front_vehicle_distance = closest_distance

    # Display the distance of the front vehicle if detected
    if front_vehicle_distance is not None:
        cv2.putText(im_bgr, f'Dist to Front Vehicle: {front_vehicle_distance:.2f} m',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Color set to black (0, 0, 0)

    # Write the frame with results to the output video
    out.write(im_bgr)

    # Display the resulting frame in real-time
    cv2.imshow('Video with Object Detection, Lane Markings, and Distance to Front Vehicle', im_bgr)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
picam2.stop()
out.release()
cv2.destroyAllWindows()
