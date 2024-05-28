import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'yolov5s', source="local")

# Function to draw grid lines and label the segments
def draw_grid(frame, rows, cols, room_area):
    frame_height, frame_width = frame.shape[:2]

    # Calculate the area of each segment
    segment_area = room_area / 2

    # Find the vertical midpoint based on the area
    left_area = 0
    mid_point = 0
    for col in range(cols):
        left_area += np.sum(frame[:, col * (frame_width // cols): (col + 1) * (frame_width // cols)])
        if left_area >= segment_area:
            mid_point = (col + 1) * (frame_width // cols)
            break

    # Draw vertical line to divide the frame into two segments
    cv2.line(frame, (mid_point, 0), (mid_point, frame_height), (255, 0, 0), 2)

    # Label each segment
    cv2.putText(frame, "Segment 1", (mid_point // 2 - 100, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(frame, "Segment 2", (mid_point + mid_point // 2 - 100, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)


# Function to calculate area given a list of contours
def calculate_area(contours):
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)
    return total_area


# Function to determine segment based on person's bounding box
def determine_segment(x_center, frame_width):
    if x_center < frame_width // 2:
        return "Segment 1"
    else:
        return "Segment 2"


# Path to your video file
video_path = '3452304387-preview.mp4'

# Initialize VideoCapture
cap = cv2.VideoCapture(video_path)

# Get the video's frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read the first frame to define the ROI (Region of Interest)
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read the video frame.")
    exit()

# Define the ROI (manually or using image processing techniques)
# For demonstration purposes, let's assume we manually define a rectangular ROI
roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]  # Example points defining a rectangular ROI
roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

# Confidence threshold for detections
confidence_threshold = 0.5

# Dictionary to store detection times for each person
detection_times = {}

# Dictionary to store detection times for each person in each segment
segment_detection_times = {"Segment 1": [], "Segment 2": []}

# Frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Variable to store room area
room_area = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Filter detections by confidence threshold
    detections = results.pred[0][results.pred[0][:, 4] > confidence_threshold]

    # Detect contours in the ROI
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area enclosed by the contours
    room_area = calculate_area(contours)

    # Update end frame for all persons
    for person_id in detection_times.keys():
        detection_times[person_id]['end_frame'] = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Process each detection
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.int().tolist()[:6]  # Convert to integers and extract first 6 elements
        if cls == 0:  # Check if detected object is a person
            person_id = hash((x1, y1, x2, y2))  # Unique identifier for each person based on bbox coordinates

            # Determine the segment
            x_center = (x1 + x2) // 2
            segment = determine_segment(x_center, frame_width)

            # Update detection times for the segment
            if person_id not in detection_times:
                detection_times[person_id] = {'start_frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                                              'end_frame': cap.get(cv2.CAP_PROP_POS_FRAMES)}
                segment_detection_times[segment].append(detection_times[person_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw grid over the frame
    draw_grid(frame, 1, 2, room_area)  # Update: 1 row, 2 columns

    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Calculate and display the time duration of detections in each segment
for segment, detection_times_list in segment_detection_times.items():
    segment_duration_seconds = sum((detection_info['end_frame'] - detection_info['start_frame']) / fps
                                   for detection_info in detection_times_list if detection_info['end_frame'] is not None)
    print(f"Duration of detections in {segment}: {segment_duration_seconds:.2f} seconds.")

# Release VideoCapture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Display room area
print(f"Room Area: {room_area} sq. pixels")
