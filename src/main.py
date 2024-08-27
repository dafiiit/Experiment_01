import cv2
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import numpy as np

# Modell und Feature Extractor laden
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Modell im Evaluierungsmodus mit aktiviertem Dropout (für Unsicherheitsmessung)
model.eval()
model.train()

# Kamera initialisieren
cap = cv2.VideoCapture(0)


# Funktion zur Vorhersage mit Unsicherheitsabschätzung
def predict_with_uncertainty(frame, num_samples=10):
    img = Image.fromarray(frame)
    inputs = feature_extractor(images=img, return_tensors="pt")
    logits = []

    # Mehrfache Vorhersagen zur Unsicherheitsabschätzung
    for _ in range(num_samples):
        outputs = model(**inputs)
        logits.append(outputs.logits.detach().numpy())

    # Berechnung der mittleren Wahrscheinlichkeiten und der Unsicherheit (Varianz)
    logits = np.array(logits)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    mean_probs = np.mean(probs, axis=0)
    uncertainty_map = np.var(probs, axis=0).max(axis=-1)

    return mean_probs, uncertainty_map


# Funktion zur Anwendung der Unsicherheitskarte auf das Bild
def apply_uncertainty_overlay(frame, uncertainty_map):
    # Unsicherheitskarte auf die Größe des Bildes skalieren
    uncertainty_map = cv2.resize(uncertainty_map, (frame.shape[1], frame.shape[0]))
    overlay = (uncertainty_map * 255).astype(np.uint8)

    # Farbkarte anwenden (von Blau (niedrig) zu Rot (hoch))
    overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.7, overlay_color, 0.3, 0)
    return result


# Hauptschleife für die Echtzeitverarbeitung
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vorhersage machen und Unsicherheitskarte berechnen
    mean_probs, uncertainty_map = predict_with_uncertainty(frame)

    # Unsicherheitskarte auf das Bild anwenden
    output_frame = apply_uncertainty_overlay(frame, uncertainty_map)

    # Bild anzeigen
    cv2.imshow("Camera with Uncertainty", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
