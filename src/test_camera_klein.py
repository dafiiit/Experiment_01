import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Testbild', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Fehler beim Erfassen des Bildes von der Kamera.")
cap.release()
