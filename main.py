import cv2
import numpy as np
import os

EMOTIONS = ["Neutro", "Felice", "Sorpresa", "Triste", "Arrabbiato", "Disgustato", "Paura", "Disprezzo"]

if not os.path.exists('emotion-ferplus-8.onnx'):
    print("ERRORE: Manca il file emotion-ferplus-8.onnx")
    exit()
if not os.path.exists('haarcascade_frontalface_default.xml'):
    print("ERRORE: Manca haarcascade_frontalface_default.xml")
    exit()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')

immagini_reazione = {}
try:
    immagini_reazione["Neutro"] = cv2.imread("neutrale.jpg")
    immagini_reazione["Arrabbiato"] = cv2.imread("rabbia.jpg")
    immagini_reazione["Felice"] = cv2.imread("felice.jpg")
    immagini_reazione["Sorpresa"] = cv2.imread("sorpreso.jpg")
    immagini_reazione["Triste"] = cv2.imread("tristezza.jpg")
except:
    pass

cap = cv2.VideoCapture(0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

while True:
    ret, frame_left = cap.read()
    if not ret: break

    frame_left = cv2.flip(frame_left, 1)
    h_cam, w_cam, _ = frame_left.shape

    frame_right = np.zeros((h_cam, w_cam, 3), dtype=np.uint8)
    frame_right[:] = (40, 40, 40) 
    
    gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emozione_dominante = ""
    confidenza = 0.0
    colore = (255, 255, 255)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame_left, (x, y), (x+w, y+h), (100, 100, 100), 2)
        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            resized_face = cv2.resize(roi_gray, (64, 64))
            blob = cv2.dnn.blobFromImage(resized_face, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            
            net.setInput(blob)
            scores = net.forward()[0]
            probs = softmax(scores)
            
            best_idx = np.argmax(probs)
            emozione_dominante = EMOTIONS[best_idx]
            confidenza = probs[best_idx]

            if emozione_dominante == "Arrabbiato": colore = (0, 0, 255)
            elif emozione_dominante == "Felice": colore = (0, 255, 0)
            elif emozione_dominante == "Sorpresa": colore = (0, 255, 255)
            elif emozione_dominante == "Triste": colore = (255, 0, 0)

            testo_cam = f"{emozione_dominante}: {int(confidenza*100)}%"
            cv2.putText(frame_left, testo_cam, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colore, 2)

            img_extra = immagini_reazione.get(emozione_dominante)
            
            if img_extra is not None and confidenza > 0.50:
                frame_right = cv2.resize(img_extra, (w_cam, h_cam))
            else:
                cv2.putText(frame_right, emozione_dominante, (50, h_cam//2), cv2.FONT_HERSHEY_SIMPLEX, 2, colore, 3)

        except Exception:
            pass

    combined_view = np.hstack((frame_left, frame_right))
    
    cv2.imshow('Emotion Split View', combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()