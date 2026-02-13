import cv2
from camera.camera_manager import CameraManager
from vision.face_detector import FaceDetector
from vision.face_embedder import FaceEmbedder
from services.attendance_service import AttendanceService
from core.logger import logger


def run():
    camera = CameraManager()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    attendance_service = AttendanceService()

    camera.initialize()

    try:
        while True:
            frame = camera.read_frame()

            detections = detector.detect(frame)

            for face in detections:
                x1, y1, x2, y2 = face["box"]

                # Crop face region
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                # Generate embedding
                embedding = embedder.get_embedding(face_crop)

                if embedding is None:
                    continue

                # Match identity
                candidate_id, score = attendance_service.match(embedding)

                label = "Unknown"

                if candidate_id:
                    label = f"{candidate_id} ({score:.2f})"
                else:
                    label = f"Unknown ({score:.2f})"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Quantum Exam Monitor - Identity Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception:
        logger.exception("System encountered an error.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
