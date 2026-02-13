import cv2
from camera.camera_manager import CameraManager
from core.logger import logger


def run():
    camera = CameraManager()
    camera.initialize()

    try:
        while True:
            frame = camera.read_frame()

            cv2.imshow("Quantum Exam Monitor - Raw Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logger.exception("System encountered an error.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
