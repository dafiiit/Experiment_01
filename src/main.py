import cv2
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import numpy as np

# Load the model and feature extractor (now using ViTImageProcessor)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Model in train mode with dropout enabled for uncertainty estimation
model.train()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Function to make a prediction with uncertainty estimation
def predict_with_uncertainty(frame, num_samples=10):
    img = Image.fromarray(frame)
    inputs = feature_extractor(images=img, return_tensors="pt")
    logits = []

    # Multiple predictions for uncertainty estimation
    for _ in range(num_samples):
        outputs = model(**inputs)
        logits.append(outputs.logits.detach().numpy())

    # Calculate mean probabilities and uncertainty (variance)
    logits = np.array(logits)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    mean_probs = np.mean(probs, axis=0)
    uncertainty_map = np.var(probs, axis=0).max(axis=-1)

    return mean_probs, uncertainty_map

# Function to apply the uncertainty map as an overlay on the image
def apply_uncertainty_overlay(frame, uncertainty_map):
    # Resize uncertainty map to the size of the frame
    uncertainty_map = cv2.resize(uncertainty_map, (frame.shape[1], frame.shape[0]))
    overlay = (uncertainty_map * 255).astype(np.uint8)

    # Apply color map (from blue (low) to red (high))
    overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.7, overlay_color, 0.3, 0)
    return result

# Main loop for real-time processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a prediction and compute the uncertainty map
    mean_probs, uncertainty_map = predict_with_uncertainty(frame)

    # Apply the uncertainty map to the frame
    output_frame = apply_uncertainty_overlay(frame, uncertainty_map)

    # Display the frame
    cv2.imshow("Camera with Uncertainty", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
