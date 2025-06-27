from wrapper import align, preprocess_from_file, write_csv

class AlignLyrics:
    def __init__(self, audio_file, lyrics_file, word_file=None, method="MTL", cuda=False):
        self.audio_file = audio_file
        self.lyrics_file = lyrics_file
        self.word_file = word_file
        self.method = method
        self.cuda = cuda

    def run(self):
        audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(
            self.audio_file, self.lyrics_file, self.word_file)

        word_align, words = align(
            audio, words, lyrics_p, idx_word_p, idx_line_p,
            method=self.method, cuda=self.cuda)
        
        resolution = 256 / 22050 * 3
        word_align = [list(map(lambda x: x * resolution, line)) for line in word_align]

        return word_align, words
    
    def write_csv(self, pred_file, word_align, words):
        write_csv(pred_file, word_align, words)

if __name__ == "__main__":
    audio_file = "EXAMPLE/data/vocals.wav"   # pre-computed source-separated vocals; These models do not work with mixture input.
    lyrics_file = "EXAMPLE/data/lyrics.txt"   # example: jamendolyrics/lyrics/*.raw.txt"
    word_file = None   # example: jamendolyrics/lyrics/*.words.txt"; Set to None if you don't have it
    method = "MTL"                             # "Baseline", "MTL", "Baseline_BDR", "MTL_BDR"
    cuda=False                                 # set True if you have access to a GPU  

    aligner = AlignLyrics(audio_file, lyrics_file, word_file, method, cuda)
    word_align, words = aligner.run()

    # pred_file = "./MTL.csv" 
    # write_csv(pred_file, word_align, words)
    # resolution = 256 / 22050 * 3
    # word_align = [list(map(lambda x: x * resolution, line)) for line in word_align]
    # for i in range(10):
    #     print("Word Boundaries: ", word_align.pop())  # Remove the first element which is the header
    #     print("Words: ", words.pop())