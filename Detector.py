import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

def auto_capture_id(model_path, output_dir, quality_threshold=0.9, save_mode="auto",
                    min_interval=2, capture_best_only=True, display_preview=True):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    # Get webcam resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Web camera resolution: {width}x{height}")

    # Tracking variables
    last_capture_time = 0
    captured_count = 0
    best_quality_ev = 0
    best_capture = None

    # Instructions
    print(f"- Auto-save when quality is {quality_threshold * 100}% or higher.")
    if capture_best_only:
        print("Capturing only the best image.")
    if save_mode == "manual":
        print("Press 's' to manually capture.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if frame capture fails

        # Create a copy of the image
        clean_frame = frame.copy()

        # Run YOLOv8 prediction
        results = model(frame, conf=0.3)  # Ignore detections below 30% confidence

        # Find the best detection
        best_det = None
        best_con = 0
        best_box = None

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                confs = boxes.conf
                best_idx = confs.argmax().item()  # Convert tensor to index
                conf = confs[best_idx].item()  # Get highest confidence score

                if conf > best_con:
                    best_con = conf
                    box = boxes[best_idx].xyxy[0].numpy()
                    best_box = [int(i) for i in box]
                    x1, y1, x2, y2 = best_box
                    best_det = clean_frame[y1:y2, x1:x2]  # Crop detected ID

        # Draw detection box & quality indicators
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Capture: {captured_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if best_quality_ev > 0:
            cv2.putText(display_frame, f"Best Quality: {best_quality_ev:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw the best bounding box
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            quality_color = (0, int(255 * best_con), int(255 * (1 - best_con)))
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), quality_color, 2)

            quality_text = f"Quality: {best_con:.2f}"
            cv2.putText(display_frame, quality_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

            status = "Good Capture" if best_con >= quality_threshold else "Move Closer"
            cv2.putText(display_frame, status, (width // 2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)

            # Auto capture logic
            current_time = time.time()
            if save_mode == "auto" and best_con >= quality_threshold and current_time - last_capture_time > min_interval:
                if capture_best_only:
                    if best_con >= best_quality_ev:
                        filename = f"{output_dir}/best_image.jpg"
                        cv2.imwrite(filename, best_det)

                        best_quality_ev = best_con
                        best_capture = best_det
                        last_capture_time = current_time
                        captured_count = 1

                        print(f"New best ID card captured (quality: {best_con:.2f})")
                else:
                    timestamp = int(current_time)
                    filename = f"{output_dir}/id_card_{timestamp}.jpg"
                    cv2.imwrite(filename, best_det)

                    last_capture_time = current_time
                    captured_count += 1

                    if best_con > best_quality_ev:
                        best_quality_ev = best_con
                        best_capture = best_det

                    print(f"ID card captured (quality: {best_con:.2f})")

        # Display preview
        if display_preview:
            cv2.imshow("Auto ID Card Detector", display_frame)

            if best_capture is not None:
                cv2.imshow("Best ID Card", best_capture)

            if best_det is not None and not capture_best_only:
                cv2.imshow("Best Detected ID Card", best_det)

        key = cv2.waitKey(1)

        if key == ord('s') and best_det is not None:
            if capture_best_only and best_con > best_quality_ev:
                filename = f"{output_dir}/best_image_manual.jpg"
                cv2.imwrite(filename, best_det)

                best_quality_ev = best_con
                best_capture = best_det
                captured_count = 1
                print(f"New best ID card saved manually (quality: {best_con:.2f})")
            elif not capture_best_only:
                timestamp = int(time.time())
                filename = f"{output_dir}/id_card_{timestamp}_manual.jpg"
                cv2.imwrite(filename, best_det)
                captured_count += 1

                if best_con > best_quality_ev:
                    best_quality_ev = best_con
                    best_capture = best_det

                print(f"ID card captured manually (quality: {best_con:.2f})")

        elif key == ord('q'):
            break  # Exit the loop on 'q' key press

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    if capture_best_only:
        print(f"Best ID card quality: {best_quality_ev:.2f}")
    else:
        print(f"Total captured ID cards: {captured_count}")
        print(f"Best ID card quality: {best_quality_ev:.2f}")
    
    print("Session Complete")


if __name__ == "__main__":
    model_path = "E:/Enadoc/Python Projects/Auto ID Card Ditector/best.pt"
    
    auto_capture_id(
        model_path=model_path,
        output_dir="captured_cards",
        quality_threshold=0.9,
        min_interval=2,
        save_mode="auto",
        display_preview=True,
        capture_best_only=True
    )
