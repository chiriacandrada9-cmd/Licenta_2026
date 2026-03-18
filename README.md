# Control PC prin Gesturi și Voce

Aplicație desktop de control al calculatorului folosind gesturi ale mâinii recunoscute prin camera web și comenzi vocale, fără conexiune la internet. Întreaga procesare se realizează local (edge computing), garantând confidențialitatea datelor biometrice.

---

## Cuprins

1. [Descriere](#descriere)
2. [Funcționalități](#funcționalități)
3. [Arhitectură](#arhitectură)
4. [Structura proiectului](#structura-proiectului)
5. [Dependențe și biblioteci](#dependențe-și-biblioteci)
6. [Cerințe sistem](#cerințe-sistem)
7. [Instalare](#instalare)
8. [Utilizare](#utilizare)
9. [Ghid gesturi](#ghid-gesturi)
10. [Comenzi vocale](#comenzi-vocale)
11. [Configurare (settings.json)](#configurare)
12. [Detalii tehnice per modul](#detalii-tehnice-per-modul)
13. [Algoritmi de procesare](#algoritmi-de-procesare)
14. [Integrare OS (Windows)](#integrare-os-windows)
15. [Suite de teste](#suite-de-teste)
16. [Compilare executabil](#compilare-executabil)
17. [Procesare locală și conformitate](#procesare-locală-și-conformitate)
18. [Decizii de proiectare](#decizii-de-proiectare)
19. [Referințe bibliografice](#referințe-bibliografice)

---

## Descriere

Proiectul implementează un sistem de interacțiune om-calculator (HCI — Human-Computer Interaction) bazat pe două modalități complementare:

1. **Viziune computerizată** — camera web capturează imagini ale mâinii utilizatorului. Un model de rețele neuronale (MediaPipe Hand Landmarker) detectează 21 de puncte de reper (landmarks) pe fiecare mână. Pe baza pozițiilor acestor puncte, un motor de clasificare a gesturilor determină acțiunea dorită (click, scroll, drag, etc.), iar coordonatele degetului arătător sunt mapate pe ecranul monitorului pentru controlul cursorului.

2. **Procesare a vorbirii** — microfonul capturează audio PCM. Un algoritm de detecție a activității vocale (WebRTC VAD) identifică segmentele de vorbire și le trimite unui model de recunoaștere automată a vorbirii (faster-whisper, bazat pe arhitectura Whisper de la OpenAI). Textul transcris este apoi analizat de un parser de comenzi care mapează fraze (în română sau engleză) la acțiuni de mouse, tastatură sau gestiune ferestre.

Toate componentele de procesare (detecția mâinii, clasificarea gesturilor, recunoașterea vorbirii) rulează **exclusiv pe CPU-ul local**, fără a trimite date la servicii cloud. Acest lucru garantează latență minimă și confidențialitatea completă a datelor biometrice.

---

## Funcționalități

### Control prin gesturi
- Mișcare cursor cu degetul arătător
- Click stânga prin ciupire (thumb + index)
- Click dreapta prin palmă deschisă
- Double click prin ciupire menținută
- Scroll vertical prin semnul V (două degete)
- Drag & Drop prin trei degete ridicate
- Comutare desktop virtual prin swipe lateral
- Click mijlociu prin semn V + thumb

### Control vocal
- 30+ comenzi vocale predefinite (RO + EN)
- Deschidere aplicații (calculator, notepad, chrome, etc.)
- Control ferestre (minimize, maximize, închide)
- Comenzi tastatură (copy, paste, undo, select all)
- Control volum și media
- Mod dictare (transformă vorbirea în text tastat)

### Caracteristici tehnice
- Procesare 100% offline (fără conexiune la internet)
- Suport DPI Awareness pentru monitoare cu scalare
- Bypass UIPI pentru focalizare fiabilă a ferestrelor
- Filtru One-Euro adaptiv pentru netezire cursor
- Buffer de stabilitate temporal pentru clasificare gesturi
- Histereză pentru prevenirea oscilației gesturilor
- Sistem de thread-uri pentru procesare paralelă
- Interfață system tray cu pictogramă

---

## Arhitectură

### Pipeline-ul de procesare

```
┌─────────────────────────────────────────────────────────────────┐
│                    HAND PIPELINE                                 │
│                                                                  │
│  Camera ──→ HandTracker ──→ GestureRecognizer ──→ Smoother      │
│  (BGR)     (MediaPipe       (clasificare +       (1€ Filter)    │
│            landmarks)       stabilitate)                         │
│                                                   ↓              │
│                              normalize_to_screen (mapare ROI)   │
│                                                   ↓              │
│                              MouseController / KeyboardController│
│                              (Win32 SendInput)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    VOICE PIPELINE                                │
│                                                                  │
│  Microphone ──→ AudioCapture ──→ VAD ──→ SpeechRecognizer       │
│  (PCM 16kHz)   (sounddevice)   (WebRTC)  (faster-whisper)       │
│                                                   ↓              │
│                              CommandParser (text → acțiune)     │
│                                                   ↓              │
│                              Orchestrator dispatch               │
│                              (Mouse / Keyboard / WindowManager) │
└─────────────────────────────────────────────────────────────────┘
```

### Model de threading

Aplicația utilizează 3 thread-uri dedicate + thread-ul principal:

| Thread | Funcție | Frecvență |
|--------|---------|-----------|
| **Main** | Afișare previzualizare OpenCV (`imshow`) | ~30 FPS |
| **Camera** | Captură video + detecție mână + clasificare gesturi | ~30 FPS |
| **Audio** | Captură audio + VAD + acumulare utterance | ~33 Hz (blocuri de 30ms) |
| **Processing** | Consumare cozi gesture/voice + dispatch acțiuni | ~120 Hz |

Comunicarea între thread-uri se face prin cozi thread-safe (`queue.Queue`):
- `gesture_queue` — gesturi clasificate de la camera thread la processing thread
- `utterance_queue` — segmente audio de la audio thread la processing thread

---

## Structura proiectului

```
├── main.py                      # Punct de intrare (argparse, DPI, tray/preview)
├── config.py                    # Gestionare configurări (load/save/deep merge)
├── settings.json                # Toți parametrii configurabili
├── requirements.txt             # Dependențe Python
│
├── modules/                     # Module funcționale
│   ├── __init__.py
│   ├── orchestrator.py          # Coordonator central (thread-uri, cozi, dispatch)
│   ├── camera.py                # Captură video OpenCV (thread-safe)
│   ├── hand_tracker.py          # Detecție mână MediaPipe Tasks API (21 landmarks)
│   ├── gestures.py              # Motor clasificare gesturi + buffer stabilitate
│   ├── mouse_controller.py      # Control mouse Win32 (SendInput, MOUSEINPUT)
│   ├── keyboard_controller.py   # Simulare tastatură Win32 (SendInput, KEYBDINPUT)
│   ├── window_manager.py        # Gestiune ferestre (EnumWindows, SetForegroundWindow)
│   ├── audio_capture.py         # Captură audio (sounddevice, PCM 16kHz mono)
│   ├── vad.py                   # Detecție activitate vocală (WebRTC VAD)
│   ├── speech_recognizer.py     # Recunoaștere vocală (faster-whisper CTranslate2)
│   ├── command_parser.py        # Parsare text → ParsedCommand (RO + EN)
│   └── tray.py                  # System tray icon (pystray)
│
├── utils/                       # Utilități
│   ├── __init__.py
│   ├── geometry.py              # Funcții matematice (distanțe, unghiuri, ROI mapping)
│   ├── one_euro_filter.py       # Filtru 1€ adaptiv (Casiez et al., CHI '12)
│   └── smoothing.py             # Smoother coordonate (1€ Filter + deadzone + outlier)
│
├── tests/                       # Teste automate (93 teste)
│   ├── __init__.py
│   ├── test_command_parser.py   # 36 teste — comenzi vocale RO + EN
│   ├── test_config.py           # 14 teste — load/save/merge configurări
│   ├── test_geometry.py         # 15 teste — distanțe, unghiuri, extensie degete, ROI
│   ├── test_gestures.py         # 20 teste — clasificare gesturi + buffer stabilitate
│   └── test_smoothing.py        #  8 teste — filtru 1€, deadzone, convergență
│
├── assets/                      # Modele ML (descărcate automat la prima rulare)
│   └── hand_landmarker.task     # Model MediaPipe (~10 MB, descărcat automat)
│
└── build/                       # Configurare pachetizare
    ├── build.spec               # Specificație PyInstaller
    └── build.bat                # Script compilare executabil
```

---

## Dependențe și biblioteci

### Biblioteci Python (requirements.txt)

| Bibliotecă | Versiune min. | Rol | Detalii |
|-----------|---------------|-----|---------|
| `opencv-python` | ≥4.9 | Captură video, afișare previzualizare | Interfață cu camera web, desenare landmarks pe imagine, `cv2.imshow()` pentru preview |
| `mediapipe` | ≥0.10 | Detecție mână cu 21 puncte de reper | Model TFLite pre-antrenat, Tasks API cu `HandLandmarker`, inferență pe CPU via XNNPACK |
| `numpy` | ≥1.26 | Operații matriciale și manipulare array-uri | Conversii BGR→RGB, buffer-e audio, operații vectoriale |
| `pywin32` | ≥306 | Acces la API-ul Windows nativ | `win32gui` (EnumWindows, SetForegroundWindow), `win32con` (constante Windows), `win32api` |
| `sounddevice` | ≥0.4 | Captură audio în timp real | Stream PCM 16-bit mono la 16000 Hz, callback non-blocant |
| `webrtcvad-wheels` | ≥2.0.10 | Detecție activitate vocală (VAD) | Algoritm WebRTC Google, 3 niveluri de agresivitate, cadre de 10/20/30ms |
| `faster-whisper` | ≥1.0 | Recunoaștere vocală offline | Reimplementare Whisper cu CTranslate2, inferență 4x mai rapidă decât original, suport limba română |
| `pystray` | ≥0.19 | Pictogramă system tray | Meniu cu opțiuni start/stop/quit, pictogramă dinamică |
| `Pillow` | ≥10.0 | Manipulare imagini pentru tray icon | Generare pictogramă dinamică, conversie formate |
| `pyinstaller` | ≥6.0 | Pachetizare executabil standalone | Generare .exe cu toate dependențele incluse |

### Biblioteci standard Python utilizate

| Modul | Utilizat în | Scop |
|-------|------------|------|
| `ctypes` | `main.py`, `mouse_controller.py`, `keyboard_controller.py` | Apeluri Win32 API (SendInput, SetProcessDpiAwareness, keybd_event) |
| `threading` | `orchestrator.py`, `camera.py` | Thread-uri paralele pentru cameră, audio, procesare |
| `queue` | `orchestrator.py` | Cozi thread-safe pentru comunicare inter-thread |
| `argparse` | `main.py` | Parsare argumente linia de comandă |
| `logging` | Toate modulele | Sistem de logging unificat |
| `time` | `gestures.py`, `one_euro_filter.py` | Cooldown-uri, timestamp-uri |
| `math` | `geometry.py`, `smoothing.py` | Funcții trigonometrice, hypot |
| `collections` | `gestures.py` | `deque` pentru buffer-e cu dimensiune fixă |
| `re` | `command_parser.py` | Expresii regulate pentru normalizare text |
| `unicodedata` | `command_parser.py` | Eliminare diacritice pentru fuzzy matching |
| `gc` | `orchestrator.py` | Apelar garbage collector la oprire (eliberare memorie modele ML) |
| `os` | `speech_recognizer.py`, `config.py`, `hand_tracker.py` | Căi fișiere, număr CPU-uri |
| `json` | `config.py` | Serializare/deserializare configurări |
| `subprocess` | `window_manager.py` | Lansare procese (deschidere aplicații) |

### API-uri externe

| API | Modul | Metode cheie |
|-----|-------|-------------|
| **MediaPipe Tasks API** | `hand_tracker.py` | `HandLandmarker.create_from_options()`, `detect()` — detecție 21 landmarks per mână |
| **Win32 API (ctypes)** | `mouse_controller.py` | `SendInput(MOUSEINPUT)` — mișcare cursor, click-uri, scroll |
| **Win32 API (ctypes)** | `keyboard_controller.py` | `SendInput(KEYBDINPUT)` — taste individuale și combinații |
| **Win32 API (pywin32)** | `window_manager.py` | `EnumWindows`, `SetForegroundWindow`, `ShowWindow`, `GetWindowText` |
| **Win32 API (ctypes)** | `main.py` | `SetProcessDpiAwareness(2)` — awareness per-monitor DPI |
| **WebRTC VAD** | `vad.py` | `Vad(aggressiveness)`, `is_speech(frame)` — detecție cadre de vorbire |
| **faster-whisper** | `speech_recognizer.py` | `WhisperModel.transcribe()` — transcriere audio → text |

---

## Cerințe sistem

| Cerință | Specificație |
|---------|-------------|
| **Sistem de operare** | Windows 10 sau 11 (64-bit) |
| **Python** | 3.11 sau mai nou |
| **Cameră web** | Orice cameră USB sau integrată |
| **Microfon** | Necesar doar pentru comenzi vocale (opțional) |
| **RAM** | Minim 4 GB (recomandat 8 GB) |
| **CPU** | Orice procesor modern x86-64 (nu necesită GPU) |
| **Spațiu disc** | ~500 MB (inclusiv modelele ML) |

---

## Instalare

### 1. Clonare repository

```bash
git clone https://github.com/cata2lin/Licenta.git
cd Licenta
```

### 2. Creare mediu virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalare dependențe

```bash
pip install -r requirements.txt
```

**Notă**: La prima rulare, modelul MediaPipe Hand Landmarker (~10 MB) se descarcă automat în directorul `assets/`.

---

## Utilizare

### Pornire aplicație

```bash
venv\Scripts\python main.py
```

Sau fără system tray:

```bash
venv\Scripts\python main.py --no-tray
```

### Argumente linia de comandă

| Argument | Valori | Implicit | Descriere |
|----------|--------|----------|-----------|
| `--mode` | `hand`, `voice`, `combined` | `combined` | Mod de funcționare |
| `--no-preview` | — | dezactivat | Dezactivează fereastra de previzualizare |
| `--no-tray` | — | dezactivat | Rulare fără pictogramă system tray |

### Oprire aplicație

- Apasă **Q** sau **Escape** pe fereastra de previzualizare (aceasta trebuie focalizată)
- **Ctrl+C** în terminal
- Click dreapta pe pictograma tray → Quit

---

## Ghid gesturi

### Gesturi statice

| Gest | Cum se face | Acțiune declanșată |
|------|-------------|-------------------|
| ☝️ **Arătare (POINT)** | Index ridicat, restul degetelor strânse | Mișcare cursor — cursorul urmărește vârful indexului |
| 🤏 **Ciupire (PINCH)** | Vârful thumb atinge vârful index | Click stânga la poziția curentă a cursorului |
| 🤏 **Ciupire menținută (PINCH_HOLD)** | Menține ciupirea ~4 cadre consecutiv | Double click |
| ✊ **Pumn (FIST)** | Toate degetele strânse | Neutru — oprește mișcarea, eliberează drag |
| 🖐️ **Palmă deschisă (PALM_OPEN)** | 4+ degete extinse (index, mijlociu, inelar + opțional degetul mic) | Click dreapta (o singură dată per intrare în gest) |
| ✌️ **Două degete (TWO_FINGERS)** | Index + mijlociu ridicate (semnul V), thumb strâns | Mod scroll — mișcare mână sus/jos derulează pagina |
| 🤟 **Trei degete (THREE_FINGERS)** | Index + mijlociu + inelar ridicate | Drag & Drop — menține click stânga și mișcă cursorul |
| 👍 **Thumb singur (THUMB_ONLY)** | Doar thumb ridicat, clar separat | Modificator (rezervat) |
| 🤙 **Peace + Thumb (PEACE_THUMB)** | Thumb + index + mijlociu ridicate | Click mijlociu (o singură dată per intrare în gest) |

### Gesturi dinamice

| Gest | Cum se face | Acțiune |
|------|-------------|---------|
| 👋 **Swipe stânga** | Mișcare rapidă laterală a încheieturii ← | Desktop virtual anterior (Win+Ctrl+←) |
| 👋 **Swipe dreapta** | Mișcare rapidă laterală a încheieturii → | Desktop virtual următor (Win+Ctrl+→) |

### Puncte de reper MediaPipe (Landmark Indices)

```
        8 — INDEX_TIP
       /
      7 — INDEX_DIP
     /
    6 — INDEX_PIP
   /
  5 — INDEX_MCP
 /
0 — WRIST ── 1(THUMB_CMC) ── 2(THUMB_MCP) ── 3(THUMB_IP) ── 4(THUMB_TIP)
 \
  9 — MIDDLE_MCP ── 10 ── 11 ── 12(MIDDLE_TIP)
   \
   13 — RING_MCP ── 14 ── 15 ── 16(RING_TIP)
     \
     17 — PINKY_MCP ── 18 ── 19 ── 20(PINKY_TIP)
```

Indici utilizați pentru clasificare: `WRIST(0)`, `THUMB_TIP(4)`, `THUMB_IP(3)`, `THUMB_MCP(2)`, `INDEX_TIP(8)`, `INDEX_PIP(6)`, `INDEX_MCP(5)`, `MIDDLE_TIP(12)`, `MIDDLE_PIP(10)`, `MIDDLE_MCP(9)`, `RING_TIP(16)`, `RING_PIP(14)`, `RING_MCP(13)`, `PINKY_TIP(20)`, `PINKY_PIP(18)`, `PINKY_MCP(17)`.

### Mecanisme anti-zgomot

| Mecanism | Descriere |
|----------|-----------|
| **Buffer de stabilitate** | Un gest trebuie detectat pe N cadre consecutive (implicit 4) înainte de a fi confirmat |
| **Histereză pinch** | Prag intrare ciupire: 0.05, prag ieșire: 0.08 — previne oscilația la limită |
| **Histereză degete** | Marja ±0.04 pentru schimbarea stării extins/strâns a fiecărui deget |
| **Cooldown click** | 300ms minim între evenimente de click succesive |
| **Cooldown swipe** | 600ms între swipe-uri consecutive |
| **Filtru One-Euro** | Netezire adaptivă — cutoff mic la repaus (stabil), cutoff mare la mișcare (responsiv) |
| **Rejecție outlier** | Salturi >15% din ecran într-un singur cadru sunt limitate (anti-teleportare) |
| **Zonă moartă** | Mișcări <0.3% din ecran sunt suprimate (anti-tremur rezidual) |
| **Fire-once** | PALM_OPEN și PEACE_THUMB declanșează acțiunea o singură dată per activare |

---

## Comenzi vocale

### Tabel complet comenzi

| Comandă (RO) | Comandă (EN) | Acțiune | Tip |
|--------------|--------------|---------|-----|
| „click", „clic", „apasa" | „click" | Click stânga | Mouse |
| „click dreapta", „clic dreapta" | „right click" | Click dreapta | Mouse |
| „dublu click", „dublu clic" | „double click" | Double click | Mouse |
| „scroll sus", „deruleaza sus" | „scroll up" | Scroll în sus | Mouse |
| „scroll jos", „deruleaza jos" | „scroll down" | Scroll în jos | Mouse |
| „copiaza" | „copy" | Ctrl+C | Tastatură |
| „lipeste" | „paste" | Ctrl+V | Tastatură |
| „anuleaza" | „undo" | Ctrl+Z | Tastatură |
| „selecteaza tot" | „select all" | Ctrl+A | Tastatură |
| „enter" | „enter" | Tasta Enter | Tastatură |
| „escape", „inchide" | „escape" | Tasta Escape | Tastatură |
| „tab" | „tab" | Tasta Tab | Tastatură |
| „minimizeaza" | „minimize" | Minimizare fereastră activă | Ferestre |
| „maximizeaza" | „maximize" | Maximizare fereastră activă | Ferestre |
| „inchide fereastra" | „close window" | Alt+F4 (închide fereastră) | Ferestre |
| „urmatoarea fereastra" | „next window" | Alt+Tab | Ferestre |
| „desktop", „arata desktop" | „show desktop" | Win+D (arată desktop) | Ferestre |
| „deschide {aplicație}" | „open {app}" | Lansare aplicație | Ferestre |
| „tasteaza {text}", „scrie {text}" | „type {text}" | Tastare text | Tastatură |
| „dictare", „incepe dictare" | „dictation", „start dictation" | Mod dictare ON | Control |
| „stop dictare", „opreste dictare" | „stop dictation" | Mod dictare OFF | Control |
| „mod voce" | „voice mode" | Comutare pe mod voce | Control |
| „mod mana" | „hand mode" | Comutare pe mod mâna | Control |
| „mod combinat" | „combined mode" | Comutare pe mod combinat | Control |
| „opreste", „pauza" | „stop", „pause" | Pauză control | Control |
| „porneste", „continua" | „start", „resume" | Reluare control | Control |

### Parsare comenzi

Parsarea comenzilor utilizează următorul algoritm:
1. Textul transcris este normalizat: lowercase, eliminare diacritice (ă→a, ț→t), colapsare spații
2. Se caută prima potrivire din tabela de comenzi (ordonate după prioritate)
3. Comenzile cu argument (ex. „deschide {app}") captează textul rămas după trigger
4. Comenzile fără potrivire returnează `UNKNOWN`

### Aplicații preconfigurate

| Alias | Executabil | Comandă |
|-------|-----------|---------|
| chrome | chrome.exe | „deschide chrome" |
| notepad | notepad.exe | „deschide notepad" |
| calculator | calc.exe | „deschide calculator" |
| explorer | explorer.exe | „deschide explorer" |
| terminal | wt.exe | „deschide terminal" |
| paint | mspaint.exe | „deschide paint" |

---

## Configurare

Toți parametrii sunt configurabili din fișierul `settings.json`. Modificările se aplică la următoarea pornire a aplicației.

### Secțiunea `camera`

```json
{
  "device_index": 0,         // Index-ul camerei (0 = prima cameră)
  "width": 640,              // Lățime captură (pixeli)
  "height": 480,             // Înălțime captură (pixeli)
  "fps": 30,                 // Cadre pe secundă
  "flip_horizontal": true    // Oglindire orizontală (efect mirror)
}
```

### Secțiunea `hand_tracking`

```json
{
  "max_hands": 1,                // Număr maxim de mâini detectate simultan
  "detection_confidence": 0.7,   // Prag confidență detecție (0-1)
  "tracking_confidence": 0.7,    // Prag confidență tracking (0-1)
  "smoothing_factor": 0.35,      // Depreciat (menținut pentru compatibilitate)
  "cursor_speed_multiplier": 1.5,// Rezervat pentru scalare viteză
  "roi_x_min": 0.12,            // Marginea stângă ROI (spațiu normalizat)
  "roi_x_max": 0.88,            // Marginea dreaptă ROI
  "roi_y_min": 0.10,            // Marginea superioară ROI
  "roi_y_max": 0.90,            // Marginea inferioară ROI
  "one_euro_min_cutoff": 2.5,   // Filtru 1€: cutoff minim (stabilitate la repaus)
  "one_euro_beta": 0.8          // Filtru 1€: coeficient viteză (reactivitate)
}
```

### Secțiunea `gestures`

```json
{
  "click_hold_frames": 4,        // Cadre menținere ciupire pentru double click
  "scroll_sensitivity": 40,     // Sensibilitate scroll
  "pinch_threshold": 0.05,      // Distanță intrare ciupire (mai mic = mai strict)
  "pinch_release_threshold": 0.08, // Distanță ieșire ciupire (histereză)
  "deadzone_radius": 0.003,     // Zonă moartă cursor (spațiu normalizat)
  "stability_frames": 4,        // Cadre necesare pentru confirmare gest
  "click_cooldown_ms": 300,     // Cooldown între click-uri (milisecunde)
  "finger_hysteresis": 0.04     // Marjă histereză pentru starea degetelor
}
```

### Secțiunea `voice`

```json
{
  "sample_rate": 16000,          // Frecvență eșantionare audio (Hz)
  "vad_aggressiveness": 2,       // Agresivitate VAD (0-3, 3 = cel mai agresiv)
  "silence_duration_ms": 800,    // Durată tăcere pentru sfârșitul unei pronunțări (ms)
  "whisper_model": "base",       // Model Whisper: tiny, base, small, medium, large
  "whisper_language": "ro",      // Limba principală (ro, en, etc.)
  "whisper_device": "cpu",       // Dispozitiv inferență (cpu sau cuda)
  "wake_word": null              // Cuvânt de activare (null = mereu activ)
}
```

### Secțiunea `window_management`

```json
{
  "excluded_processes": ["explorer.exe"],  // Procese excluse de la focus
  "app_aliases": {                         // Mapare nume → executabil
    "chrome": "chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "terminal": "wt.exe",
    "paint": "mspaint.exe"
  }
}
```

### Secțiunea `app`

```json
{
  "mode": "combined",         // Mod implicit: hand, voice, combined
  "show_preview": true,       // Afișare fereastră previzualizare
  "start_minimized": false,   // Pornire minimizat
  "log_level": "INFO"         // Nivel logging: DEBUG, INFO, WARNING, ERROR
}
```

---

## Detalii tehnice per modul

### main.py — Punct de intrare

- Parsare argumente (argparse): `--mode`, `--no-preview`, `--no-tray`
- Apel `SetProcessDpiAwareness(2)` pentru awareness DPI per-monitor
- Inițializare sistem de logging (fișier + consolă)
- Pornire orchestrator
- Buclă principală de previzualizare pe thread-ul principal (necesar `cv2.imshow` pe Windows)

### modules/orchestrator.py — Coordonator central

- Inițializare și pornire a tuturor subsistemelor
- Gestiune 3 thread-uri (cameră, audio, procesare)
- Buclă de procesare gesturi: drenare coadă și acționare doar pe ultimul gest
- Buclă de procesare voce: transcriere + parsare + execuție comandă
- Metoda `stop()` include `gc.collect()` pentru eliberarea memoriei modelelor ML
- Metoda `_smooth_and_map()` aplică filtrul 1€ și maparea ROI → coordonate ecran

### modules/camera.py — Captură video

- Wrapper thread-safe peste `cv2.VideoCapture`
- Métoda `get_frame()` returnează ultimul cadru sau `None`
- Suportă configurare rezoluție, FPS, flip orizontal

### modules/hand_tracker.py — Detecție mână

- Wrapper peste `mediapipe.tasks.vision.HandLandmarker`
- Descărcare automată model la prima rulare (`hand_landmarker.task`)
- Returnează `HandData(landmarks, handedness, annotated_frame)`
- 21 puncte de reper cu coordonate (x, y, z) normalizate 0-1
- Metodă `draw_landmarks()` pentru desenare pe frame

### modules/gestures.py — Clasificare gesturi

- Clasificare statică bazată pe starea extins/strâns a fiecărui deget
- Detecție ciupire cu histereză (praguri diferite intrare/ieșire)
- Buffer de stabilitate: necesită N cadre identice pentru confirmare
- Detecție swipe bazată pe velocitatea orizontală a încheieturii
- Histereză per deget: marjă ±0.04 pentru prevenirea flickering-ului
- Guard pinch/point: verificare distanță thumb-index pentru dezambiguizare
- Guard thumb/fist: verificare `thumb_index_spread` pentru separare

### modules/mouse_controller.py — Control mouse

- Utilizează `ctypes.windll.user32.SendInput` cu structuri `MOUSEINPUT`
- Flag `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE` pentru poziționare absolută
- Normalizare coordonate la spațiul virtual 0-65535
- Metode: `move_to()`, `left_click()`, `right_click()`, `double_click()`, `middle_click()`, `scroll()`, `left_down()`, `left_up()`

### modules/keyboard_controller.py — Simulare tastatură

- Utilizează `ctypes.windll.user32.SendInput` cu structuri `KEYBDINPUT`
- Metode: `press_key()`, `release_key()`, `type_text()`, `hotkey()`
- Suportă taste virtuale (VK_RETURN, VK_ESCAPE, etc.) și combinații (Ctrl+C, ALT+F4)
- Metodă specială `type_text()` pentru tastare caracter cu caracter

### modules/window_manager.py — Gestiune ferestre

- `EnumWindows` pentru listare ferestre vizibile
- `SetForegroundWindow` cu bypass UIPI (keybd_event sintetic ALT)
- `ShowWindow` pentru minimize/maximize/restore
- Lansare aplicații prin `subprocess.Popen`
- Filtrare procese excluse (configurat în settings)

### modules/audio_capture.py — Captură audio

- Stream `sounddevice.RawInputStream` la 16000 Hz, mono, 16-bit PCM
- Callback non-blocant: acumulează audio în buffer
- Cadre de 30ms (480 samples) compatibile cu WebRTC VAD

### modules/vad.py — Detecție activitate vocală

- Instanță `webrtcvad.Vad` cu agresivitate configurabilă (0-3)
- Cadre de 30ms analizate individual
- Pre-roll: include 300ms de audio dinainte de detectarea vorbirii
- Tăcere: marchează sfârșitul pronunțării după 800ms
- Returnează segmentul audio complet al pronunțării

### modules/speech_recognizer.py — Recunoaștere vocală

- `WhisperModel` din faster-whisper cu backend CTranslate2
- Model configurat: `base` (performanță/acuratețe optimă)
- Limitare `cpu_threads = min(4, os.cpu_count())` contra overhead-ului
- Anti-halucinare: filtrare transcrieri goale sau repetitive
- Returnează text transcris sau string gol

### modules/command_parser.py — Parsare comenzi

- Normalizare text: lowercase + eliminare diacritice + colapsare spații
- Tabel de 30+ triggers cu acțiuni mapper
- Primul match câștigă (ordonate după specificitate)
- Suport argumente: `deschide {app}`, `tasteaza {text}`
- Fallback: `ActionType.UNKNOWN` pentru text nerecunoscut

### utils/geometry.py — Funcții geometrice

- `distance_2d(a, b)` — distanța euclidiană 2D
- `angle_between(a, b, c)` — unghi format de 3 puncte (radiani)
- `is_finger_extended(tip, pip, mcp)` — deget extins dacă tip este mai departe de mcp decât pip
- `normalize_to_screen(x, y, w, h, roi)` — mapare din spațiu normalizat 0-1 la pixeli ecran prin Regiune de Interes

### utils/one_euro_filter.py — Filtru One-Euro

- Implementare completă a filtrului 1€ (Casiez et al., CHI 2012)
- `LowPassFilter` — filtru trece-jos de ordinul 1 cu alpha configurabil
- `OneEuroFilter` — filtru adaptiv: cutoff = min_cutoff + beta × |velocitate|
- Suport timestamp opțional pentru frecvență de eșantionare dinamică

### utils/smoothing.py — Smoother coordonate

- Backend: două instanțe `OneEuroFilter` (axa X și Y)
- Rejecție outlier: limitare salturi >15% per cadru
- Zonă moartă: suprimare mișcări <0.3% din ecran
- Filtrul primește TOATE eșantioanele (chiar și în deadzone) pentru estimare precisă a vitezei

---

## Algoritmi de procesare

### Filtrul One-Euro (1€ Filter)

Filtrul One-Euro este un filtru trece-jos adaptiv proiectat pentru netezirea semnalelor în timp real în sisteme de interacțiune om-calculator. A fost publicat de Casiez, Roussel și Vogel la conferința CHI 2012.

**Principiul de funcționare:**

Frecvența de tăiere (cutoff) este adaptată dinamic pe baza vitezei semnalului de intrare:

```
cutoff = min_cutoff + beta × |velocitate_filtrată|
```

- Când mâna este **statică** sau se mișcă **lent**: velocitatea este mică → cutoff aproape de `min_cutoff` (valoare mică) → netezire puternică → cursor stabil, fără tremur
- Când mâna se mișcă **rapid**: velocitatea este mare → cutoff crește proporțional cu beta → netezire redusă → tracking responsiv, fără lag

**Parametri:**

| Parametru | Valoare | Efect |
|-----------|---------|-------|
| `min_cutoff` | 2.5 Hz | Stabilitate la repaus (mai mic = mai stabil dar mai laggy) |
| `beta` | 0.8 | Reactivitate la mișcare (mai mare = mai responsiv la viteză) |
| `d_cutoff` | 1.0 Hz | Netezire estimare viteza (de obicei fix) |

**Formula alpha (coeficient de netezire):**

```
tau = 1 / (2π × cutoff)
te  = 1 / freq
alpha = 1 / (1 + tau/te)
```

La 30 FPS cu `min_cutoff=2.5`: alpha ≈ 0.34 (34% din valoarea nouă per cadru la repaus).

### Mapare ROI (Regiune de Interes)

Coordonatele mâinii (normalizate 0-1 de MediaPipe) sunt mapate pe ecranul monitorului printr-o regiune de interes configurabilă:

```
screen_x = clamp((hand_x - roi_x_min) / (roi_x_max - roi_x_min)) × screen_width
screen_y = clamp((hand_y - roi_y_min) / (roi_y_max - roi_y_min)) × screen_height
```

Implicit, ROI = (0.12, 0.88) × (0.10, 0.90), adică utilizatorul folosește ~76% × 80% din câmpul vizual al camerei pentru a acoperi întregul ecran.

### Clasificare degete (is_finger_extended)

Un deget non-thumb este considerat **extins** dacă vârful (TIP) este mai departe de încheietura (MCP) decât articulația intermediară (PIP), cu o marjă de histereză:

```
dist(tip, mcp) > dist(pip, mcp) + margin
```

Pentru degetul mare (thumb), se verifică distanța orizontală (x) față de MCP, ținând cont de handedness (mâna stângă/dreaptă):
- Mâna dreaptă: thumb extins dacă tip.x < mcp.x - margin (MediaPipe: imagine mirrored)
- Mâna stângă: thumb extins dacă tip.x > mcp.x + margin

---

## Integrare OS (Windows)

### DPI Awareness

Aplicația apelează `SetProcessDpiAwareness(2)` (Per-Monitor DPI Awareness v1) la pornire prin `ctypes.windll.shcore`. Acest lucru asigură că coordonatele cursorului calculate sunt corecte pe monitoare cu scalare (125%, 150%, 175%, etc.). Fără această setare, coordonatele ar fi aplicate la rezoluția virtuală scalată, rezultând un offset al cursorului.

### UIPI Bypass

User Interface Privilege Isolation (UIPI) în Windows poate bloca apelul `SetForegroundWindow` dacă procesul nostru nu este în prim-plan. Soluția implementată: înainte de `SetForegroundWindow`, se trimite o secvență sintetică de taste ALT (keybd_event) care „deblochează" mecanismul de focalizare al Windows-ului.

### Simulare input via SendInput

Toate acțiunile de mouse și tastatură sunt simulate prin API-ul Win32 `SendInput` (via ctypes), nu prin funcții de nivel înalt. Acest lucru oferă:
- Compatibilitate cu DirectX și aplicații fullscreen
- Funcționare în jocuri și aplicații cu input raw
- Permisiuni corecte la nivel de kernel

---

## Suite de teste

### Rulare teste

```bash
venv\Scripts\python -m pytest tests/ -v
```

### Acoperire (93 teste totale)

| Fișier test | Număr teste | Ce verifică |
|------------|------------|-------------|
| `test_command_parser.py` | 36 | Toate comenzile vocale (RO + EN), cazuri limită, insensitivitate la majuscule, eliminare diacritice |
| `test_config.py` | 14 | Încărcare/salvare configurări, deep merge, setări parțiale, JSON corupt |
| `test_geometry.py` | 15 | Distanțe, unghiuri, extensie degete, mapare ROI, clamping coordonate |
| `test_gestures.py` | 20 | Toate gesturile statice, buffer stabilitate, rejecție zgomot, histereză |
| `test_smoothing.py` | 8 | Filtru 1€, zonă moartă, convergență, reducere jitter, tracking rapid |

Toate testele folosesc date sintetice — nu necesită cameră web sau microfon.

---

## Compilare executabil

```bash
cd build
build.bat
```

Se generează un executabil standalone în `build/dist/` care include:
- Toate modulele Python
- Modelul MediaPipe
- Runtime-ul CTranslate2 (pentru faster-whisper)
- Biblioteci native (OpenCV, numpy, etc.)

Executabilul nu necesită Python instalat pe mașina destinație.

### Excluderi PyInstaller

Pentru reducerea dimensiunii, sunt excluse: matplotlib, pandas, scipy, IPython, notebook, PyQt5, tensorboard, pytest.

---

## Procesare locală și conformitate

Toate datele sunt procesate **exclusiv pe mașina locală**. Nu se transmit informații la servicii cloud.

### Avantaje privire la confidențialitate

| Context reglementar | Risc cloud | Avantajul procesării locale |
|---------------------|-----------|---------------------------|
| **GDPR** (Protecția datelor UE) | Trimiterea audio în afara jurisdicției necesită consimțământ explicit și audituri | Datele biometrice nu sunt colectate permanent; procesate în RAM volatil, eliminate după transcriere |
| **HIPAA** (Date medicale SUA) | Dictarea medicală necesită contracte BAA (Business Associate Agreement) | Nu implică terți; datele rămân pe stația de lucru securizată |
| **SOC 2** (Securitate corporativă) | Transferul fișierelor crește suprafața de atac prin rețele publice | Control suveran; niciun risc de interceptare audio |

### Cazuri de utilizare în industrie

- **Chirurgie** — medicii manipulează imagini 3D MRI/PACS prin gesturi, menținând câmpul steril
- **Manufacturare** — operatorii cu mănuși de protecție controlează panouri industriale prin gesturi (ecranele capacitive nu funcționează cu mănuși groase)
- **Reparații auto** — mecanicii navighează scheme electrice prin voce în timp ce lucrează sub vehicule
- **Juridic** — dictare confidențială fără expunere cloud (transcriere offline conformă GDPR)
- **Accesibilitate** — persoane cu mobilitate redusă (artrită, leziuni spinale) controlează PC-ul prin micro-gesturi cu praguri ROI ajustabile

---

## Decizii de proiectare

1. **Procesare edge (offline)** — privacitate completă, latență sub 100ms, nu depinde de conexiune internet. Trade-off: nu poate folosi modele cloud mai puternice (GPT-4, Gemini).

2. **ctypes.SendInput vs pyautogui** — `SendInput` oferă control de nivel kernel peste mouse și tastatură, funcționând corect în DirectX, jocuri și aplicații cu UAC. `pyautogui` folosește API-uri de nivel mai înalt care pot fi blocate.

3. **Preview pe thread-ul principal** — `cv2.imshow()` trebuie apelat pe thread-ul principal pe Windows pentru randare fiabilă a ferestrei. Thread-ul de cameră stochează frame-urile annotate într-o variabilă thread-safe.

4. **Filtrul One-Euro vs EMA** — EMA simplu are compromisul fix jitter-vs-lag. Filtrul 1€ adaptează dinamic frecvența de tăiere pe baza vitezei, oferind simultan stabilitate la repaus și reactivitate la mișcare.

5. **Histereză gesturi** — praguri diferite de intrare/ieșire previn oscilația rapidă la valori limită (Schmitt trigger software).

6. **Drenare coadă gesturi** — thread-ul de procesare consumă TOATE gesturile din coadă la fiecare iterație și acționează doar pe ultimul. Previne acumularea de gesturi vechi și lag-ul de intrare.

7. **Fire-once pentru click-uri** — PALM_OPEN și PEACE_THUMB declanșează acțiunea o singură dată la intrarea în gest. Previne click-uri repetate în timp ce gestul este menținut.

8. **gc.collect() la oprire** — apel explicit al garbage collector-ului la oprirea aplicației pentru eliberarea memoriei ocupate de modelele ML (MediaPipe, Whisper ~300MB).

---

## Referințe bibliografice

1. **Casiez, G., Roussel, N., Vogel, D.** (2012). „1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in Interactive Systems". *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12)*. ACM, pp. 2527-2530. DOI: 10.1145/2207676.2208639. URL: https://gery.casiez.net/1euro/

2. **Lugaresi, C., Tang, J., Nash, H., et al.** (2019). „MediaPipe: A Framework for Building Perception Pipelines". *arXiv preprint arXiv:1906.08172*. URL: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

3. **Radford, A., Kim, J.W., Xu, T., et al.** (2023). „Robust Speech Recognition via Large-Scale Weak Supervision". *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*. URL: https://github.com/openai/whisper

4. **SYSTRAN/faster-whisper** — Reimplementare CTranslate2 a modelului Whisper cu inferență de 4x mai rapidă. URL: https://github.com/SYSTRAN/faster-whisper

5. **WebRTC Voice Activity Detector** — Algoritm de detecție activitate vocală dezvoltat de Google. URL: https://webrtc.org/

6. **Zhang, F., Bazarevsky, V., Vakunov, A., et al.** (2020). „MediaPipe Hands: On-device Real-time Hand Tracking". *arXiv preprint arXiv:2006.10214*.

7. **Microsoft Win32 API — SendInput Function**. URL: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput

8. **Microsoft Win32 API — SetProcessDpiAwareness**. URL: https://learn.microsoft.com/en-us/windows/win32/api/shellscalingapi/nf-shellscalingapi-setprocessdpiawareness
