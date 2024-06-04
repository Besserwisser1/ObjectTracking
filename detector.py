import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from random import randrange

from SFSORT import SFSORT


model = YOLO('yolov8m.pt', 'detect')

# Check for GPU availability
device = select_device('0')
# Devolve the processing to selected devices
model.to(device)


# Load the video file
cap = cv2.VideoCapture('/content/Sample.mp4')

# Get the frame rate, frame width, and frame height
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the MP4 codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

# Organize tracker arguments into standard format
tracker_arguments = {"dynamic_tuning": True, "cth": 0.7,
                      "high_th": 0.82, "high_th_m": 0.1,
                      "match_th_first": 0.5, "match_th_first_m": 0.05,
                      "match_th_second": 0.1, "low_th": 0.3,
                      "new_track_th": 0.7, "new_track_th_m": 0.1,
                      "marginal_timeout": (7 * frame_rate // 10),
                      "central_timeout": frame_rate,
                      "horizontal_margin": frame_width // 10,
                      "vertical_margin": frame_height // 10,
                      "frame_width": frame_width,
                      "frame_height": frame_height}

# Instantiate a tracker
tracker = SFSORT(tracker_arguments)

# Define a color list for track visualization
colors = {}

# Process each frame of the video
while cap.isOpened():
   # Load the frame
  ret, frame = cap.read()
  if not ret:
      break

  # Detect people in the frame
  prediction = model.predict(frame, imgsz=(800,1440), conf=0.1, iou=0.45,
                              half=False, device=device, max_det=99, classes=0,
                              verbose=False)

  # Exclude additional information from the predictions
  prediction_results = prediction[0].boxes.cpu().numpy()

  # Update the tracker with the latest detections
  tracks = tracker.update(prediction_results.xyxy, prediction_results.conf)

  # Skip additional analysis if the tracker is not currently tracking anyone
  if len(tracks) == 0:
      continue

  # Extract tracking data from the tracker
  bbox_list = tracks[:, 0]
  track_id_list = tracks[:, 1]

  # Visualize tracks
  for idx, (track_id, bbox) in enumerate(zip(track_id_list, bbox_list)):
    # Define a new color for newly detected tracks
    if track_id not in colors:
        colors[track_id] = (randrange(255), randrange(255), randrange(255))

    color = colors[track_id]

    # Extract the bounding box coordinates
    x0, y0, x1, y1 = map(int, bbox)

    # Draw the bounding boxes on the frame
    annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    # Put the track label on the frame alongside the bounding box
    cv2.putText(annotated_frame, str(track_id), (x0, y0-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


  # Write the frame to the output video file
  out.write(annotated_frame)

# Release everything when done
cap.release()
out.release()