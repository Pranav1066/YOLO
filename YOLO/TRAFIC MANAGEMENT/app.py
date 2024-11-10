import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import torch
import time
import numpy as np

# Check if a CUDA-compatible GPU is available and load the model on it
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov10n.pt").to(device)

# Traffic light timing for each video
traffic_light_durations = [15, 21, 56, 56]  # Green or red light duration in seconds for each video

# Threshold for determining light color based on car count
max_car_count = 25  # Adjust threshold as needed

# Hypothetical scale factor to convert pixel movement to meters (for speed in km/h)
pixel_to_meter_scale = 0.05  # Adjust based on actual camera setup

# Streamlit UI for uploading and processing videos
st.title("Traffic Light Control with YOLO (Real-Time Display)")
st.write("Upload four videos, and the model will predict whether each traffic light should be red or green, and "
         "display speed estimates in km/h.")

# File upload widgets for four videos
uploaded_videos = [st.file_uploader(f"Upload video {i+1}", type=["mp4", "avi", "mov", "mkv"]) for i in range(4)]

# Loop over each uploaded video
for i, uploaded_video in enumerate(uploaded_videos):
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Open video using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        # Initialize total car count and frame count for this video
        total_car_count = 0
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Streamlit placeholder for displaying frames
        stframe = st.empty()  # Placeholder for displaying frames

        # Store previous frame positions for speed estimation
        prev_positions = {}

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video is finished

            # Move the frame to GPU if needed
            if device == "cuda":
                frame = torch.from_numpy(frame).to(device)

            # Run YOLO detection on the frame
            results = model(frame)

            # Initialize car count for this frame
            frame_car_count = 0

            # Track current positions of objects for speed calculation
            current_positions = {}

            # Loop through each detection result
            for result in results:
                for j, box in enumerate(result.boxes):
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    if class_name in ["car", "truck"]:
                        frame_car_count += 1  # Increment car count if car or truck detected

                        # Draw bounding box and label
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]  # Convert to Python int
                        label = f"{class_name} ({int(box.conf * 100)}%)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Calculate the center position of the bounding box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        current_positions[j] = (center_x, center_y)

                        # Calculate speed if previous position exists
                        if j in prev_positions:
                            prev_x, prev_y = prev_positions[j]
                            distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                            distance_meters = distance_pixels * pixel_to_meter_scale
                            speed_kmh = (distance_meters * 3.6) * fps  # Speed in km/h
                            label_speed = f"Speed: {speed_kmh:.2f} km/h"
                            cv2.putText(frame, label_speed, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Update positions for the next frame
            prev_positions = current_positions

            # Update total car count and frame count for the video
            total_car_count += frame_car_count
            frame_count += 1

            # Convert frame to CPU and back to numpy if on GPU
            if device == "cuda":
                frame = frame.cpu().numpy()

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame with bounding boxes in Streamlit
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Calculate the average car count per frame for the video
        avg_car_count = total_car_count / frame_count if frame_count > 0 else 0

        # Determine if the light should be red or green for this video
        light_color = "Red" if avg_car_count >= max_car_count else "Green"
        duration = traffic_light_durations[i]

        # Display result for this video
        st.write(f"Video {i+1} Recommendation: {light_color} light for {duration} seconds")
        st.write(f"Average car count per frame: {avg_car_count:.2f}")

        # Release the video capture object
        cap.release()
    else:
        st.warning(f"Please upload video {i+1} to begin processing.")
