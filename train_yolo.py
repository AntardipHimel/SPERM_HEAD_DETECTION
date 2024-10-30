from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model with some modifications
    model.train(
        data='C:/Users/rayre/sperm_head_detection/data.yaml',  # Use forward slashes or double backslashes
        epochs=500,           # Adjusted number of training epochs (start with fewer, increase if needed)
        imgsz=1280,            # Image size
        batch=8,              # Batch size (adjust based on your GPU's memory)
        lr0=0.01,             # Learning rate (adjust depending on how fast/slow the loss decreases)
        name='sperm_detection',  # Experiment name
        project='runs/train',  # Where to save the results (default is runs/train)
        cache=True,           # Cache images for faster training
        augment=True,         # Use data augmentation
        patience=50,          # Enable early stopping to prevent overfitting
        device=0              # GPU ID (set to 0 if you have one GPU, or 'cpu' for CPU)
    )

    print("Training completed!")