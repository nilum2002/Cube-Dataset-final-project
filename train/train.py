from ultralytics import YOLO

# # # yolo detect train \
# # model=yolov8n.pt \
# # data=data.yaml \
# # epochs=100 \
# # freeze=10 \
# # lr0=0.001 \
# # mixup=0.2 \
# # weight_decay=0.0005 \
# # patience=20

# Load the nano model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/media/nilum/New_Volume/01.projects/Dataset-cubes/Cube-Detection-Dataset-3/data.yaml', 
    epochs=100, 
    imgsz=640, 
    device='0', 
    optimizer="AdamW",
    lr0=5e-4, 
    project='/media/nilum/New_Volume/01.projects/Dataset-cubes/runs', 
    single_cls = True, 
    freeze=10,
    weight_decay=5e-4,
    mixup=0.2) # Use device='cpu' if no GPU

######################################################
#              To Implement - Warmup ratio           #
######################################################
# warmup ratio
# waight decay
# learning scheduler - cosine 


model = YOLO("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train-9/weights/best.pt")

metrics = model.val(data="/media/nilum/New_Volume/01.projects/Dataset-cubes/Cube-Detection-Dataset-3/data.yaml", split="val")

print(metrics)