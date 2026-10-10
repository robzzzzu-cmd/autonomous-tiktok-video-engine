#!/usr/bin/env python3
"""
Royalty-Free Cozy Lofi Acoustic Background Loop Synthesizer.
Synthesizes a warm, soothing 16-bit 44.1kHz audio loop (Cmaj7 - Am7 - Dm7 - G7)
using only Python's standard library. Zero external dependencies required.
"""

import math
import os
import struct
import wave

def synthesize_lofi_track(output_wav_path: str = "assets/audio/lofi_acoustic_loop.wav", duration_seconds: float = 24.0):
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sample_rate = 44100
    num_samples = int(sample_rate * duration_seconds)

    # Warm chord progression: Cmaj7 -> Am7 -> Dm7 -> G7
    chord_notes = [
        [130.81, 196.00, 246.94, 329.63], # Cmaj7
        [110.00, 164.81, 196.00, 261.63], # Am7
        [146.83, 220.00, 261.63, 349.23], # Dm7
        [98.00, 146.83, 174.61, 246.94]   # G7
    ]

    chord_duration = duration_seconds / len(chord_notes)
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        chord_idx = int((t % duration_seconds) // chord_duration) % len(chord_notes)
        chord = chord_notes[chord_idx]
        t_chord = t % chord_duration

        # Soft acoustic decay envelope
        env = math.exp(-t_chord * 0.45) * 0.4 + 0.15

        # Harmonics
        val_f = 0.0
        for freq in chord:
            val_f += math.sin(2.0 * math.pi * freq * t) * 0.35
            val_f += math.sin(2.0 * math.pi * freq * 2.0 * t) * 0.12
            val_f += math.sin(2.0 * math.pi * freq * 3.0 * t) * 0.04

        # Vinyl warm crackle
        crackle = ((math.sin(t * 1234.5) * 43758.5453) % 1.0 - 0.5) * 0.015

        val = (val_f * env + crackle) * 0.35
        val = max(-0.95, min(0.95, val))
        samples.append(int(val * 32767.0))

    with wave.open(output_wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        packed = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(packed)

    print(f"Synthesized lofi background loop saved to: {output_wav_path}")

if __name__ == "__main__":
    synthesize_lofi_track()
