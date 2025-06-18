from wrapper import align, preprocess_from_file, write_csv

audio_file = "data/vocals.wav"   # pre-computed source-separated vocals; These models do not work with mixture input.
lyrics_file = "data/lyrics.txt"   # example: jamendolyrics/lyrics/*.raw.txt"
word_file = None   # example: jamendolyrics/lyrics/*.words.txt"; Set to None if you don't have it
method = "MTL"                             # "Baseline", "MTL", "Baseline_BDR", "MTL_BDR"
cuda=False                                 # set True if you have access to a GPU

pred_file = "./MTL.csv"   

audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(audio_file, lyrics_file, word_file)

word_align, words = align(audio, words, lyrics_p, idx_word_p, idx_line_p, method=method, cuda=False)

write_csv(pred_file, word_align, words)
