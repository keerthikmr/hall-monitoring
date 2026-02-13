import cv2
from camera.camera_manager import CameraManager
from vision.face_detector import FaceDetector
from core.logger import logger


def run():
    camera = CameraManager()
    detector = FaceDetector()

    camera.initialize()

    try:
        while True:
            frame = camera.read_frame()

            detections = detector.detect(frame)

            for face in detections:
                x1, y1, x2, y2 = face["box"]
                confidence = face["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Quantum Exam Monitor - Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception:
        logger.exception("System encountered an error.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
