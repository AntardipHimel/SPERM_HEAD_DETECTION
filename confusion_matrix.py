import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend for matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r'C:\Users\rayre\sperm_head_detection\runs\train\sperm_detection82\weights\best.pt')

# Run inference on your validation set
# Run inference and save predicted images
results = model(r'C:\Users\rayre\sperm_head_detection\images\split_images_val_jpg', conf=0.1, save=True)

# This will save the images with bounding boxes in runs/detect/exp/

# Initialize lists to store true and predicted labels
true_labels = []
pred_labels = []

# Loop over the results to gather true and predicted labels
for result in results:
    if result.boxes:  # Check if any boxes were detected
        true_label = result.boxes.cls.cpu().numpy()  # Ground truth class labels
        pred_label = result.boxes.conf.argmax(dim=1).cpu().numpy()  # Predicted class labels with highest confidence
        
        # Append true and predicted labels to the lists
        true_labels.append(true_label)
        pred_labels.append(pred_label)

# Flatten the lists of true and predicted labels
if true_labels and pred_labels:  # Ensure there are labels before proceeding
    true_labels = np.concatenate(true_labels).flatten()
    pred_labels = np.concatenate(pred_labels).flatten()

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Print the confusion matrix for debugging
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Sperm Detection')

    # Save the confusion matrix to a known location
    save_path = r'C:\Users\rayre\sperm_head_detection'
    plt.savefig(save_path)
    print(f"Confusion matrix saved at: {save_path}")

    # Display the confusion matrix
    plt.show()
else:
    print("No detections were made on the validation set.")
