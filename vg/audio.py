
import logging
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy
import hashlib
import soundfile as sf


def extract_mfcc(f, truncate=None):
    #logging.info("Extracting features from {}".format(f))
    try:
        (sig, rate) = sf.read(f)
    except:
        logging.warning("Error reading file {}".format(f))
        return None      
    try:
        if truncate is not None:
            max_len = truncate*rate
            mfcc_feat = psf.mfcc(sig[:max_len], rate)
        else:
            mfcc_feat = psf.mfcc(sig, rate)
        return np.asarray(mfcc_feat, dtype='float32')

    except:
        logging.warning("Error extracting features from file {}".format(f))
        return None


def encode(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def load_audio(texts, audio_dir):
    """Load audio from audio_dir.
    """
    logging.info("Loading audio")
    for text in texts:
        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), "rb") as au:
            yield au.read()
