import cv2
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import numpy as np

# Modell und Feature-Extractor laden
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Modell im Train-Modus belassen, um Dropout für Unsicherheitsmessung zu aktivieren
model.train()

# Kamera initialisieren und ein Bild aufnehmen
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Bild speichern oder weiterverarbeiten
if ret:
    # Vorhersage mit Unsicherheitsabschätzung
    def predict_with_uncertainty(image, num_samples=10):
        img = Image.fromarray(image)
        inputs = feature_extractor(images=img, return_tensors="pt")
        logits = []

        for _ in range(num_samples):
            outputs = model(**inputs)
            logits.append(outputs.logits.detach().numpy())

        logits = np.array(logits)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        mean_probs = np.mean(probs, axis=0)
        uncertainty_map = np.var(probs, axis=0).max(axis=-1)

        return mean_probs, uncertainty_map

    # Vorhersage und Unsicherheitskarte berechnen
    mean_probs, uncertainty_map = predict_with_uncertainty(frame)

    # Unsicherheitskarte auf die Größe des Bildes skalieren
    uncertainty_map_resized = cv2.resize(uncertainty_map, (frame.shape[1], frame.shape[0]))
    overlay = (uncertainty_map_resized * 255).astype(np.uint8)

    # Farbkarte anwenden
    overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    output_frame = cv2.addWeighted(frame, 0.7, overlay_color, 0.3, 0)

    # Bild anzeigen
    cv2.imshow("Image with Uncertainty", output_frame)
    cv2.waitKey(0)  # Auf Tastendruck warten
    cv2.destroyAllWindows()

else:
    print("Fehler beim Erfassen des Bildes von der Kamera.")
