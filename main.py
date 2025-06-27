import crepe
from scipy.io import wavfile
import csv

from EXAMPLE import AlignLyrics
from midi import TimeToMidi

audio_file = "EXAMPLE/data/vocals.wav"   # pre-computed source-separated vocals; These models do not work with mixture input.
lyrics_file = "EXAMPLE/data/lyrics.txt"   # example: jamendolyrics/lyrics/*.raw.txt"
word_file = None   # example: jamendolyrics/lyrics/*.words.txt"; Set to None if you don't have it
method = "MTL"                             # "Baseline", "MTL", "Baseline_BDR", "MTL_BDR"
cuda=False                                 # set True if you have access to a GPU  

aligner = AlignLyrics(audio_file, lyrics_file, word_file, method, cuda)
word_align, words = aligner.run()
print("Lyrics alignment completed")
# aligher.write_csv("MTL.csv", word_align, words)

sr, audio = wavfile.read(audio_file)
time, frequency, confidence, activation = crepe.predict(audio, sr, model_capacity='large', viterbi=True)
print("Pitch detection completed")
# Save time, frequency, confidence, and activation to a CSV file
# output_file = '/Users/zhangboya/Desktop/pitch_data.csv'
# with open(output_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Time', 'Frequency', 'Confidence'])  # Header row
#     for t, f, c in zip(time, frequency, confidence):
#         writer.writerow([t, f, c])
frequency_dict = [[f,  c] for f, c in zip(frequency, confidence)]

midi = TimeToMidi(bpm=70, ticks_per_beat=480)
midi.toMIDI(word_align, words, frequency_dict)
midi.save_midi("output.mid")
