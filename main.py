import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Config:
    ONNX_MODEL: str = 'emotion-ferplus-8.onnx'
    CASCADE_MODEL: str = 'haarcascade_frontalface_default.xml'
    
    EMOTIONS: Tuple[str, ...] = (
        "Neutro", "Felice", "Sorpresa", "Triste", 
        "Arrabbiato", "Disgustato", "Paura", "Disprezzo"
    )
    
    IMAGE_MAPPING: Dict[str, str] = None
    
    CAMERA_INDEX: int = 0
    FLIP_HORIZONTAL: bool = True
    
    FACE_SCALE_FACTOR: float = 1.3
    FACE_MIN_NEIGHBORS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.50
    
    WINDOW_NAME: str = 'Emotion Split View'
    BACKGROUND_COLOR: Tuple[int, int, int] = (40, 40, 40)
    BBOX_COLOR: Tuple[int, int, int] = (100, 100, 100)
    BBOX_THICKNESS: int = 2
    TEXT_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE: float = 0.8
    TEXT_THICKNESS: int = 2
    LARGE_TEXT_SCALE: float = 2.0
    LARGE_TEXT_THICKNESS: int = 3
    
    EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.IMAGE_MAPPING is None:
            self.IMAGE_MAPPING = {
                "Neutro": "neutrale.jpg",
                "Arrabbiato": "rabbia.jpg",
                "Felice": "felice.jpg",
                "Sorpresa": "sorpreso.jpg",
                "Triste": "tristezza.jpg"
            }
        
        if self.EMOTION_COLORS is None:
            self.EMOTION_COLORS = {
                "Arrabbiato": (0, 0, 255),
                "Felice": (0, 255, 0),
                "Sorpresa": (0, 255, 255),
                "Triste": (255, 0, 0),
                "Neutro": (255, 255, 255),
                "default": (255, 255, 255)
            }


class EmotionDetector:
    
    def __init__(self, config: Config):
        self.config = config
        self.face_cascade = None
        self.emotion_net = None
        self._load_models()
    
    def _load_models(self) -> None:
        if not os.path.exists(self.config.CASCADE_MODEL):
            raise FileNotFoundError(
                f"ERRORE: File {self.config.CASCADE_MODEL} non trovato!\n"
                "Scaricalo da: https://github.com/opencv/opencv/tree/master/data/haarcascades"
            )
        
        self.face_cascade = cv2.CascadeClassifier(self.config.CASCADE_MODEL)
        if self.face_cascade.empty():
            raise ValueError(f"ERRORE: Impossibile caricare {self.config.CASCADE_MODEL}")
        
        if not os.path.exists(self.config.ONNX_MODEL):
            raise FileNotFoundError(
                f"ERRORE: File {self.config.ONNX_MODEL} non trovato!\n"
                "Scaricalo da: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus"
            )
        
        try:
            self.emotion_net = cv2.dnn.readNetFromONNX(self.config.ONNX_MODEL)
        except Exception as e:
            raise ValueError(f"ERRORE: Impossibile caricare {self.config.ONNX_MODEL}: {e}")
        
        print("✓ Modelli caricati con successo")
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def detect_emotion(self, gray_face: np.ndarray) -> Tuple[str, float]:
        try:
            resized_face = cv2.resize(gray_face, (64, 64))
            
            blob = cv2.dnn.blobFromImage(
                resized_face, 
                scalefactor=1.0, 
                size=(64, 64),
                mean=(0, 0, 0), 
                swapRB=False, 
                crop=False
            )
            
            self.emotion_net.setInput(blob)
            scores = self.emotion_net.forward()[0]
            
            probs = self._softmax(scores)
            
            best_idx = np.argmax(probs)
            emotion = self.config.EMOTIONS[best_idx]
            confidence = float(probs[best_idx])
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Errore durante il rilevamento emozione: {e}")
            return "Sconosciuto", 0.0


class ImageManager:
    
    def __init__(self, config: Config):
        self.config = config
        self.images: Dict[str, Optional[np.ndarray]] = {}
        self._load_images()
    
    def _load_images(self) -> None:
        loaded_count = 0
        
        for emotion, filename in self.config.IMAGE_MAPPING.items():
            if os.path.exists(filename):
                try:
                    img = cv2.imread(filename)
                    if img is not None:
                        self.images[emotion] = img
                        loaded_count += 1
                    else:
                        print(f"⚠ Impossibile caricare {filename}")
                except Exception as e:
                    print(f"⚠ Errore caricando {filename}: {e}")
            else:
                self.images[emotion] = None
        
        print(f"✓ Caricate {loaded_count}/{len(self.config.IMAGE_MAPPING)} immagini di reazione")
    
    def get_image(self, emotion: str) -> Optional[np.ndarray]:
        return self.images.get(emotion)


class EmotionApp:
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = EmotionDetector(config)
        self.image_manager = ImageManager(config)
        self.cap = None
        self._init_camera()
    
    def _init_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"ERRORE: Impossibile aprire la webcam (index={self.config.CAMERA_INDEX})\n"
                "Prova a cambiare CAMERA_INDEX in Config (es. 0, 1, 2...)"
            )
        
        print("✓ Webcam inizializzata")
    
    def _create_right_frame(
        self, 
        shape: Tuple[int, int, int],
        emotion: str,
        confidence: float,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        h, w, _ = shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = self.config.BACKGROUND_COLOR
        
        if confidence > self.config.CONFIDENCE_THRESHOLD:
            img = self.image_manager.get_image(emotion)
            if img is not None:
                try:
                    return cv2.resize(img, (w, h))
                except Exception as e:
                    print(f"Errore ridimensionamento immagine: {e}")
        
        text = emotion if emotion else "Nessuna emozione"
        text_size = cv2.getTextSize(
            text, 
            self.config.TEXT_FONT, 
            self.config.LARGE_TEXT_SCALE, 
            self.config.LARGE_TEXT_THICKNESS
        )[0]
        
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(
            frame, 
            text, 
            (text_x, text_y),
            self.config.TEXT_FONT,
            self.config.LARGE_TEXT_SCALE,
            color,
            self.config.LARGE_TEXT_THICKNESS
        )
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.FACE_SCALE_FACTOR,
            minNeighbors=self.config.FACE_MIN_NEIGHBORS
        )
        
        emotion = ""
        confidence = 0.0
        color = self.config.EMOTION_COLORS["default"]
        
        if len(faces) > 0:
            (x, y, w_face, h_face) = faces[0]
            
            cv2.rectangle(
                frame,
                (x, y),
                (x + w_face, y + h_face),
                self.config.BBOX_COLOR,
                self.config.BBOX_THICKNESS
            )
            
            roi_gray = gray[y:y + h_face, x:x + w_face]
            emotion, confidence = self.detector.detect_emotion(roi_gray)
            
            color = self.config.EMOTION_COLORS.get(
                emotion, 
                self.config.EMOTION_COLORS["default"]
            )
            
            text = f"{emotion}: {int(confidence * 100)}%"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                self.config.TEXT_FONT,
                self.config.TEXT_SCALE,
                color,
                self.config.TEXT_THICKNESS
            )
        
        frame_right = self._create_right_frame(
            (h, w, 3),
            emotion,
            confidence,
            color
        )
        
        return np.hstack((frame, frame_right))
    
    def run(self) -> None:
        print("\n" + "="*60)
        print("🎭 AI EMOTION TRIGGER - AVVIATO")
        print("="*60)
        print("Premi 'Q' per uscire")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠ Impossibile leggere dalla webcam")
                    break
                
                combined_view = self.process_frame(frame)
                cv2.imshow(self.config.WINDOW_NAME, combined_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Applicazione chiusa dall'utente")
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠ Interruzione da tastiera")
        except Exception as e:
            print(f"\n❌ ERRORE: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Risorse rilasciate")


def main():
    try:
        config = Config()
        app = EmotionApp(config)
        app.run()
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📚 Consulta il README.md per le istruzioni di setup")
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
