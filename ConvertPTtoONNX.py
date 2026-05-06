# 1. Install the Ultralytics package
!pip install ultralytics

# 2. Export the model to ONNX with Jetson-safe settings
!yolo export model=yolov8n.pt format=onnx opset=12 simplify=True
