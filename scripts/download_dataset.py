from roboflow import Roboflow

rf = Roboflow(api_key="7UD8kxPwNHTVsFQekK72")
project = rf.workspace("vehicle-mscoco").project("vehicles-coco")
dataset = project.version(1).download("coco", location="data/raw")