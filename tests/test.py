# from ultralytics import YOLO
# import cv2

# model = YOLO("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train/weights/best.pt")

# # Open video
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error opening video file")
#     exit()

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Run inference
#     results = model(frame, conf=0.9)

#     # Draw detections
#     annotated_frame = results[0].plot()

#     # Show output
#     cv2.imshow("YOLOv8 Detection", annotated_frame)

#     # Press q to exit
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()

# cv2.destroyAllWindows()



from ultralytics import YOLO
import cv2

# Load model
model = YOLO("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train-9/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # Linux backend

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Optional camera size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # YOLO inference
    results = model.predict(frame, conf=0.6, verbose=False)

    # Draw boxes
    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()