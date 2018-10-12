
import logging
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
import hashlib
import soundfile as sf
import tts
import os.path
import StringIO

def extract_mfcc(f, truncate=None):
    #logging.info("Extracting features from {}".format(f))
    (sig, rate) = sf.read(f)
    if truncate is not None:
        max_len = truncate*rate
        mfcc_feat = psf.mfcc(sig[:max_len], rate, nfft=1024)
    else:
        mfcc_feat = psf.mfcc(sig, rate, nfft=1024)
        return np.asarray(mfcc_feat, dtype='float32')



def encode(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def load_audio(texts, audio_dir):
    """Load audio from audio_dir.
    """
    logging.info("Loading audio")
    for text in texts:
        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), "rb") as au:
            yield StringIO.StringIO(au.read())

def audio_paths(texts, audio_dir):
    """Return a list of audio file paths for texts.
    """
    return [ "{}/{}.wav".format(audio_dir, encode(text)) for text in texts ]




def save_audio(texts, audio_dir):
    """Synthesize and save audio."""
    logging.info("Storing wav files")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    for text in texts:
        logging.info("Synthesizing audio for {}".format(text))
        audio = tts.synthesize(text)
        logging.info("Storing audio for {}".format(text))
        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), 'w') as out:
            out.write(audio)

