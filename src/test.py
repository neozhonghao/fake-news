from src.inference import get_prediction, initialize, LemmaTokenizer
orig_texts, real, fake, undecided, histo = get_prediction("is this real?", 10, initialize())
print("Original text: %s, real: %.5f, fake: %.5f, undecided: %d, histo: %s" % (orig_texts[0], real, fake, undecided, histo))