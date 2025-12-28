# 🎭 AI Emotion Trigger (Split View)

Questo progetto è un'applicazione di visione artificiale in tempo reale che rileva le emozioni facciali e reagisce mostrando immagini associate.

L'interfaccia è divisa in **Split-View** (Schermo diviso):
* **Sinistra (Input):** Feed della webcam con bounding box facciale, emozione rilevata e percentuale di confidenza.
* **Destra (Output):** Visualizza dinamicamente un'immagine (o meme) corrispondente all'emozione rilevata (es. *Felicità, Rabbia, Sorpresa*).

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green?style=flat&logo=opencv)

## 📋 Caratteristiche
* Rilevamento facciale rapido usando **Haar Cascades**.
* Riconoscimento emozioni avanzato usando una Rete Neurale Profonda (**ONNX FER+ Model**).
* Supporto per 8 emozioni: *Neutro, Felice, Sorpresa, Triste, Arrabbiato, Disgustato, Paura, Disprezzo*.
* Gestione errori robusta (non crasha se mancano immagini).

## 📂 Struttura del Repository

Assicurati che la tua cartella contenga questi file fondamentali:

| File | Descrizione |
| :--- | :--- |
| `emozioni_split.py` | **Script Principale**. Avvia la webcam e l'interfaccia split-screen. |
| `haarcascade_frontalface_default.xml` | Modello leggero per trovare la posizione del viso. |
| `emotion-ferplus-8.onnx` | Il "cervello" AI (Deep Learning) che classifica le emozioni. |
| `rabbia.jpg`, `felice.jpg`, ... | Immagini di reazione (personalizzabili). |

---

## 🛠️ Installazione e Setup

### 1. Clona o scarica il progetto
Scarica i file in una cartella locale.

### 2. Installa le dipendenze
Hai bisogno di Python 3 installato. Esegui questo comando nel terminale:

```bash
pip install opencv-python numpy
