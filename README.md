An AI bot for Vampire Survivors that automates character movement using computer vision and a YOLO object detection model.
The bot analyzes the game screen in real-time, detects enemies, collectibles, and hazards, predicts enemy movement, and decides optimal movement with a kite system.


Features:
Real-time screen capture using mss
Object detection with YOLO


Detects:
Enemies
Experience items / runes / coins


Pros:
Enemy movement prediction

Kite mechanic for optimal distance management

Crowd and predicted position avoidance

Attraction to safe collectibles

Smooth WASD input control

Visual debug overlay with vectors and object highlights


Pause with P.

Exit visualization with ESC.


Dependencies

Requirements
pip install -r requirements.txt

Game window resolution  must match game_area in the code (you can make it fit by looking in debug window and resizing game in window mode)


Configuration

Key parameters:
CONF_THRESHOLD = 0.45

KITE_MIN_DIST = 115

KITE_MAX_DIST = 185

RUNE_ATTRACT_FORCE = 1600.0

LOCAL_THREAT_LIMIT = 0.0012

You can try to change these parameters for better result

⚠️ Disclaimer
For research, or learning purposes only.
Use in online games may violate game rules.
