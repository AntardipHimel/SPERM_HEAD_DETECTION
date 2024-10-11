
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(
        data='C:/Users/rayre/sperm_head_detection/data.yaml',  # Use forward slashes or double backslashes
        epochs=100,            # Number of training epochs
        imgsz=640,             # Image size
        batch=4,              # Batch size (adjust based on your GPU's memory)
        name='sperm_detection',  # Experiment name
        project='runs/train',   # Where to save the results (default is runs/train)
        device=0               # GPU ID (set to 0 if you have one GPU, or 'cpu' for CPU)
    )

    print("Training completed!")

