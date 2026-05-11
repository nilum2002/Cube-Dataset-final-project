from ultralytics import YOLO

# # Load the nano model
model = YOLO('yolov8n.pt')



# # # Train the model
# results = model.train(
#     data='/media/nilum/New_Volume/01.projects/Dataset-cubes/Cube-Detection-Dataset-3/data.yaml', 
#     epochs=100, 
#     warmup_epochs=10,
#     warmup_momentum=0.9,
#     warmup_bias_lr=0.01,
#     imgsz=640, 
#     device='0', 
#     lr0=1e-7,
#     project='/media/nilum/New_Volume/01.projects/Dataset-cubes/runs', 
#     single_cls = True,
#     # freeze=10,
#     weight_decay=1e-4,
#     mixup=0.2,
#     cos_lr = True)
 

# # # Tune the model
results = model.tune(
    data='/home/intellisense05/Nilum/Cube-Dataset-final-project/Robo-Games-5/data.yaml', 
    epochs=100, 
    warmup_epochs=10,
    warmup_momentum=0.9,
    warmup_bias_lr=0.01,
    imgsz=640, 
    device='0', 
    lr0=1e-7,
    project='/home/intellisense05/Nilum/Cube-Dataset-final-project/runs2', 
    single_cls = True,
    # freeze=10,
    weight_decay=1e-4,
    mixup=0.2,
    cos_lr = True)

######################################################
#             Model Validation                       #
######################################################


# model = YOLO("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train-16/weights/best.pt")

# metrics = model.val(data="/media/nilum/New_Volume/01.projects/Dataset-cubes/Cube-Detection-Dataset-3/data.yaml", split="val", save_dir = "/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/val-15")

# print(metrics)