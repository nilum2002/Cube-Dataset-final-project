from ultralytics import YOLO

# Load the nano model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/media/nilum/New_Volume/01.projects/Dataset-cubes/Robo-Games-4/data.yaml', 
    epochs=100, 
    warmup_epochs=10,
    warmup_momentum=0.9,
    warmup_bias_lr=0.01,
    imgsz=640, 
    device='0', 
    lr0=5e-4, 
    project='/media/nilum/New_Volume/01.projects/Dataset-cubes/runs', 
    single_cls = True, 
    freeze=10,
    weight_decay=5e-4,
    mixup=0.2,
    cos_lr = True)
 

######################################################
#              To Implement - Warmup ratio           #
######################################################
# warmup ratio
# waight decay
# learning scheduler - cosine 



######################################################
#             Model Validation                       #
######################################################


model = YOLO("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train-9/weights/best.pt")

metrics = model.val(data="/media/nilum/New_Volume/01.projects/Dataset-cubes/Robo-Games-4/data.yaml", split="val")

print(metrics)