
################################################################################
#                   Dataset Setup for Older version - Roboflow                 #
################################################################################


# from roboflow import Roboflow
# rf = Roboflow(api_key="ReeIH4uTzRmmKoEc5UgQ")
# project = rf.workspace("botzillaiesl-robo-games").project("cube-detection-dataset")
# version = project.version(3)
# dataset = version.download("yolov8")
                

################################################################################
#                   New Dataset Setup for Older version - Roboflow             #
################################################################################

from roboflow import Roboflow
rf = Roboflow(api_key="ReeIH4uTzRmmKoEc5UgQ")
project = rf.workspace("botzillaiesl-robo-games").project("robo-games")
version = project.version(4)
dataset = version.download("yolov8")