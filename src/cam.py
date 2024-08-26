import time
from picamera2 import Picamera2
from libcamera import controls
import cv2


class RaspberryPiCamera:
    def __init__(self):
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_still_configuration())
        self.camera.start()
        time.sleep(2)  # Wait for camera to initialize

    def capture_image(self, filename="image.jpg"):
        self.camera.capture_file(filename)
        return filename

    def set_resolution(self, width, height):
        config = self.camera.create_still_configuration(main={"size": (width, height)})
        self.camera.configure(config)

    def set_exposure_mode(self, mode):
        self.camera.set_controls({"AeMode": mode})

    def set_awb_mode(self, mode):
        self.camera.set_controls({"AwbMode": mode})

    def set_iso(self, iso):
        self.camera.set_controls({"AnalogueGain": iso})

    def set_shutter_speed(self, speed):
        self.camera.set_controls({"ExposureTime": speed})

    def close(self):
        self.camera.close()


# Example usage
if __name__ == "__main__":
    # Initialize the camera
    pi_cam = RaspberryPiCamera()

    # Set some parameters (optional)
    pi_cam.set_resolution(1920, 1080)
    pi_cam.set_exposure_mode(controls.AeModeEnum.Auto)
    pi_cam.set_awb_mode(controls.AwbModeEnum.Auto)

    # Capture an image
    image_file = pi_cam.capture_image("test_image.jpg")

    # Display the image
    img = cv2.imread(image_file)
    cv2.imshow("Captured Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Close the camera
    pi_cam.close()
