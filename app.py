import streamlit as st
import cv2
import numpy as np
import csv
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
MODEL_PATH = 'fish_detection_model.pt'
model = YOLO(MODEL_PATH)

# Streamlit app
st.title("Fish Detection and Size Estimation")
st.write("Upload multiple images to detect fish and estimate their sizes.")

# Upload images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"])

if uploaded_files:
    # Temporary directory to save uploaded images
    temp_dir = tempfile.TemporaryDirectory()
    image_paths = []

    # Save uploaded images locally for processing
    for uploaded_file in uploaded_files:
        image_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(image_path)

    st.write("Processing images...")

    # Initialize tracking for the best image
    max_fish_count = 0
    best_image_path = None

    # Find the best image with the maximum number of fish detected
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # YOLO detection
        results = model(image)
        fish_count = len(results[0].boxes)  # Count detected fish

        if fish_count > max_fish_count:
            max_fish_count = fish_count
            best_image_path = image_path

    if best_image_path:
        st.write(f"Best Image: {os.path.basename(best_image_path)}")
        st.write(f"Number of Fish Detected: {max_fish_count}")

        # Process the best image
        best_image = cv2.imread(best_image_path)
        results = model(best_image)

        # Extract fish sizes
        fish_sizes = {}
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, [box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]])
            fish_width = x2 - x1
            fish_height = y2 - y1
            fish_size_px = np.sqrt(fish_width ** 2 + fish_height ** 2)
            fish_size_cm = fish_size_px * 0.05  # Assuming 1 pixel = 0.05 cm calibration factor

            fish_sizes[idx + 1] = fish_size_cm  # Use 1-based indexing for Fish IDs

        # Save results to CSV
        csv_output = "fish_size_summary.csv"
        with open(csv_output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Fish ID", "Size (cm)"])
            for fish_id, size in fish_sizes.items():
                writer.writerow([fish_id, f"{size:.2f}"])

        # Display results
        st.write("Fish Sizes Detected:")
        st.dataframe(
            [{"Fish ID": fish_id, "Size (cm)": f"{size:.2f}"} for fish_id, size in fish_sizes.items()]
        )

        # Download link for CSV
        with open(csv_output, "rb") as f:
            st.download_button(
                label="Download Fish Size Data (CSV)",
                data=f,
                file_name="fish_size_summary.csv",
                mime="text/csv"
            )

    else:
        st.error("No fish detected in any image.")

    # Clean up temporary files
    temp_dir.cleanup()
else:
    st.info("Please upload one or more images to start processing.")
