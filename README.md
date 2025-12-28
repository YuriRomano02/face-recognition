# 🎭 AI Emotion Trigger (Split View)

Questo progetto è un'applicazione di visione artificiale in tempo reale che rileva le emozioni facciali e reagisce mostrando immagini associate.

L'interfaccia è divisa in **Split-View** (Schermo diviso):
* **Sinistra (Input):** Feed della webcam con bounding box facciale, emozione rilevata e percentuale di confidenza.
* **Destra (Output):** Visualizza dinamicamente un'immagine (o meme) corrispondente all'emozione rilevata (es. *Felicità, Rabbia, Sorpresa*).

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green?style=flat&logo=opencv)

---

## 📋 Caratteristiche

* Rilevamento facciale rapido usando **Haar Cascades**.
* Riconoscimento emozioni avanzato usando una Rete Neurale Profonda (**ONNX FER+ Model**).
* Supporto per 8 emozioni: *Neutro, Felice, Sorpresa, Triste, Arrabbiato, Disgustato, Paura, Disprezzo*.
* Gestione errori robusta (non crasha se mancano immagini).
* Visualizzazione split-screen con feedback visivo in tempo reale.

---

## 📂 Struttura del Repository

Assicurati che la tua cartella contenga questi file fondamentali:

| File | Descrizione |
| :--- | :--- |
| `main.py` | **Script Principale**. Avvia la webcam e l'interfaccia split-screen. |
| `haarcascade_frontalface_default.xml` | Modello leggero per trovare la posizione del viso. |
| `emotion-ferplus-8.onnx` | Il "cervello" AI (Deep Learning) che classifica le emozioni. |
| `rabbia.jpg`, `felice.jpg`, `sorpreso.jpg`, `tristezza.jpg`, `neutrale.jpg` | Immagini di reazione (personalizzabili). |

---

## 🛠️ Installazione e Setup

### Prerequisiti

* **Python 3.7 o superiore** installato sul tuo sistema
* Una webcam funzionante

### 1. Clona o scarica il progetto

Scarica i file in una cartella locale oppure clona il repository:

```bash
git clone <url-repository>
cd ai-emotion-trigger
```

### 2. Installa le dipendenze

Questo progetto richiede principalmente **OpenCV** e **NumPy**. Puoi installarli usando pip:

```bash
pip install opencv-python numpy
```

#### Dettagli sulle librerie:

* **opencv-python**: La libreria principale per computer vision. Include il supporto per il modulo DNN (Deep Neural Networks) e la gestione della webcam.
* **numpy**: Utilizzato per operazioni matematiche e manipolazione degli array.

#### Installazione alternativa con requirements.txt

Se preferisci, puoi creare un file `requirements.txt` con il seguente contenuto:

```
opencv-python>=4.5.0
numpy>=1.19.0
```

E poi installare tutto con:

```bash
pip install -r requirements.txt
```

### 3. Scarica i modelli necessari

#### a) Haar Cascade (Rilevamento volti)

Scarica il file `haarcascade_frontalface_default.xml` da:
* [Repository ufficiale OpenCV](https://github.com/opencv/opencv/tree/master/data/haarcascades)

Oppure puoi scaricarlo direttamente con questo comando:

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

#### b) Modello ONNX per emozioni

Scarica il file `emotion-ferplus-8.onnx` da:
* [ONNX Model Zoo - Emotion FERPlus](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus)

### 4. Aggiungi le immagini di reazione

Crea o scarica le seguenti immagini nella stessa cartella dello script:

* `neutrale.jpg`
* `rabbia.jpg`
* `felice.jpg`
* `sorpreso.jpg`
* `tristezza.jpg`

Puoi usare qualsiasi immagine o meme che preferisci! Il programma funzionerà comunque anche senza queste immagini, mostrando solo il testo dell'emozione.

---

## 🚀 Utilizzo

### Avvio dell'applicazione

Esegui lo script principale:

```bash
python main.py
```

### Controlli

* **Q**: Premi il tasto `q` per uscire dall'applicazione.

### Come funziona

1. La webcam si attiva e cattura il tuo volto in tempo reale.
2. L'algoritmo di rilevamento facciale trova la tua faccia nel frame.
3. Il modello di deep learning analizza l'espressione facciale.
4. L'emozione rilevata viene mostrata con:
   * Testo e percentuale di confidenza sul frame sinistro
   * Immagine corrispondente sul frame destro (se disponibile e confidenza > 50%)

---

## 🎨 Personalizzazione

### Cambiare le immagini di reazione

Modifica le righe nel codice dove vengono caricate le immagini:

```python
immagini_reazione["Neutro"] = cv2.imread("la_tua_immagine.jpg")
immagini_reazione["Arrabbiato"] = cv2.imread("la_tua_immagine_rabbia.jpg")
# ... e così via
```

### Aggiungere altre emozioni

Il modello supporta 8 emozioni. Puoi aggiungere immagini per tutte:

* Disgustato
* Paura
* Disprezzo

Basta aggiungerle nel dizionario `immagini_reazione` seguendo lo stesso pattern.

### Modificare la soglia di confidenza

Di default, l'immagine appare solo se la confidenza è > 50%. Puoi modificare questa soglia nella riga:

```python
if img_extra is not None and confidenza > 0.50:
```

---

## 🐛 Risoluzione Problemi

### La webcam non si avvia

* Verifica che la webcam sia collegata e funzionante
* Prova a cambiare l'indice della camera: `cv2.VideoCapture(1)` invece di `0`

### Errore "Manca il file emotion-ferplus-8.onnx"

* Assicurati di aver scaricato il modello ONNX e che sia nella stessa cartella dello script

### Errore durante l'installazione di OpenCV

Se riscontri problemi con `opencv-python`, prova la versione headless:

```bash
pip install opencv-python-headless
```

Oppure, su Linux, potresti dover installare dipendenze aggiuntive:

```bash
sudo apt-get install python3-opencv
```

### Le immagini non vengono visualizzate

* Controlla che i file immagine esistano nella cartella corretta
* Verifica che i nomi dei file corrispondano esattamente a quelli nel codice
* Il programma continuerà a funzionare mostrando solo il testo dell'emozione

---

## 📝 Note Tecniche

* Il modello FER+ è stato addestrato sul dataset FER2013 e raggiunge buone performance nel riconoscimento delle emozioni di base.
* L'utilizzo di Haar Cascades per il rilevamento facciale è molto veloce ma meno preciso rispetto a metodi più moderni (es. DNN face detector). Per maggiore accuratezza, considera di sostituirlo con un detector basato su deep learning.
* La funzione softmax viene applicata agli output del modello per ottenere probabilità normalizzate.

---

## 📄 Licenza

Questo progetto è rilasciato per scopi educativi e di sperimentazione.

---

## 🤝 Contributi

Sentiti libero di fare fork del progetto e proporre miglioramenti!

---

## 📧 Contatti

Per domande o suggerimenti, apri una Issue nel repository.

---

**Divertiti a esplorare il mondo delle emozioni artificiali! 🎭🤖**
