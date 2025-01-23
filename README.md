Music Generator with Tension, Valence, and Energy

This repository contains a multi-parameter, emotion-driven music generator using an LSTM-based model. It allows dynamic control over:
	•	Valence (bright/dark tonal quality, e.g., major vs. minor)
	•	Tension (harmonic complexity, dissonance, chord extensions)
	•	Energy (rhythmic density, velocity, overall intensity)

Features
	1.	Seed Presets
	•	Choose from multiple preset seeds (e.g., pentatonic, rest style, minor dark) to kick-start generation.
	2.	Parameter Injection
	•	Adjust valence, tension, and energy in real time to alter chord type, note density, dynamics, or mode.
	3.	Coherence Under Control
	•	The LSTM ensures melodic and harmonic continuity even when emotional parameters change on the fly.
	4.	MIDI Output
	•	Generated music can be saved as a .mid file for playback or further editing.

How to Run
	1.	Clone or Download this repository.
	2.	Dependencies:
	•	Python 3.8+
	•	TensorFlow/Keras
	•	music21
	•	numpy, json, etc.
	•	(Optional) pip install -r requirements.txt if provided.
	3.	Prepare Model & Mapping
	•	Ensure model.h5 and mapping.json are in the repo directory (or specify correct paths).
	4.	Run Script

python main.py

	•	You’ll be prompted to select a preset seed and input valence/tension/energy values (0.0–1.0).
	•	The system generates and saves a MIDI file reflecting these parameters.

Quick Example

python main.py
# e.g.:
#  Please select a preset seed (1~7): 2
#  Enter valence (0.0~1.0): 0.5
#  Enter tension (0.0~1.0): 0.7
#  Enter energy (0.0~1.0): 0.3

A .mid file will be created, such as mel_val_0.50_ten_0.70_ene_0.30_seed2.mid, which you can open in any MIDI player or DAW.

Notes
	•	This project is experimental; abrupt transitions may occur with large parameter jumps.
	•	Feel free to explore or adjust the mapping functions (e.g., chord extensions, pitch offsets) for different creative results.

Enjoy creating and exploring expressive, user-guided music!
