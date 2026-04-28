from roboflow import Roboflow
rf = Roboflow(api_key="ReeIH4uTzRmmKoEc5UgQ")
project = rf.workspace("botzillaiesl-robo-games").project("cube-detection-dataset")
version = project.version(3)
dataset = version.download("yolov8")
                