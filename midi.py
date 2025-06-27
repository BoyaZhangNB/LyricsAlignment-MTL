import mido
from mido import Message, MidiFile, MidiTrack
from librosa import hz_to_midi
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt


# ====== Utils =====
def dur2ticks(note_length, bpm=120, ticks_per_beat=480): 
    """
    Convert note length in seconds to ticks.
    Args:
        note_length (seconds): Length of the note in seconds.
        ticks_per_beat (int): Number of ticks per beat (default is 480).
        bpm (int): Beats per minute (default is 120).

    print(dur2ticks(0.5)) should print the number of ticks for a 0.5 second note at 120 BPM, 480 ticks
    """
    return int(note_length * ticks_per_beat * bpm / 60)

def toNote(frequency, confidence):
    """
    Convert frequency to MIDI note number.
    Args:
        frequency (List[float]): Frequency in Hz.
        confidence (List[float]): Confidence level of the frequency detection (not used in this function).
    Returns:
        int: MIDI note number over a period of time.
    """

    print(frequency, confidence)

    if not frequency:
        return 0
    if frequency <= 0:
        return 0  # Return 0 for non-positive frequencies
    if len(frequency) != len(confidence):
        raise ValueError("Frequency and confidence lists must have the same length.")

    total_confidence = sum(confidence)
    normalized_confidence = [c / total_confidence for c in confidence]
    
    weighted_frequency = sum(f * c for f, c in zip(frequency, normalized_confidence)) / sum(normalized_confidence)

    return round(hz_to_midi(weighted_frequency))

def gmm_find_clusters(ts, max_clusters=6):

    ts = medfilt(ts, kernel_size=8)

    ts = np.array(ts).reshape(-1, 1)
    
    best_gmm = None
    lowest_bic = np.inf
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, n_init=5).fit(ts)
        bic = gmm.bic(ts)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    labels = best_gmm.predict(ts)
    return labels, best_gmm.n_components


class TimeToMidi:
    def __init__(self, bpm=120, ticks_per_beat=480):
        self.bpm = bpm
        self.ticks_per_beat = ticks_per_beat
        self.midi_file = MidiFile(ticks_per_beat=ticks_per_beat)

    def toMIDI(self, word_align, words, pitch): 
        # Frequency is a dictionary with time as key and frequency and confidence as value
        track = MidiTrack()
        self.midi_file.tracks.append(track)

        current_time = 0
        for i, (start, end) in enumerate(word_align):
            word = words[i]

            startidx = int(start / 0.01)  # Round to nearest 0.01 seconds
            endidx = int(end / 0.01)

            if endidx > len(pitch):
                print(f"Warning: End time {end} exceeds pitch data length {len(pitch)}. Skipping word '{word}'.")
                continue

            frequency = np.array([pitch[startidx:endidx][0][0]])  # Get the frequency data for the word
            confidence = np.array([pitch[startidx:endidx][0][1]])  # Get the confidence data for the word

            note = toNote(frequency, confidence)

            # Create a new track for each word

            print(start, end, word, note)
            # Add note on and note off messages
            track.append(Message('note_on', note=note, velocity=64, time=dur2ticks(start-current_time, self.bpm, self.ticks_per_beat)))
            track.append(Message('note_off', note=note, velocity=64, time=dur2ticks(end-start, self.bpm, self.ticks_per_beat)))
            current_time = end

    def save_midi(self, output_file):
        self.midi_file.save(output_file)

if __name__ == "__main__":
    # Example usage
    word_align = [
        (0.0, 0.0132),
        (0.0132, 0.0223),
        (0.0223, 0.0350),
        (0.0350, 0.0480),
        (0.0480, 0.0600),
        (0.0600, 0.0720),
    ]  # Example word alignments
    words = ["Hello", "world", "I", "am", "very", "happy"]  # Example words
    pitch = [
        [440, 0.9],  # Example pitch data: [frequency, confidence]
        [440, 0.8],
        [440, 0.85],
        [440, 0.95],
        [440, 0.9],
        [440, 0.88],
        [440, 0.92],
        [440, 0.91],
        [440, 0.93],
        [440, 0.94],
        [440, 0.95],
        [440, 0.96],
    ]

    midi = TimeToMidi(bpm=120, ticks_per_beat=480)
    midi.toMIDI(word_align, words, pitch)
    midi.save_midi("output.mid")