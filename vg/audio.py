
import logging
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
import hashlib
import soundfile as sf
import pydub
import tts
import os.path
import StringIO

def read_mp3(f):
    seg = pydub.AudioSegment.from_mp3(f)
    rate, sig = wav.read(StringIO.StringIO(seg.export(StringIO.StringIO(), format='wav').getvalue()))
    return (sig, rate)

def extract_mfcc(f, truncate=None, format='wav', accel=False, nfft=512):
    #logging.info("Extracting features from {}".format(f))
    if format == 'mp3':
        (sig, rate) = read_mp3(f)
    else:    
        (sig, rate) = sf.read(f)

    if truncate is not None:
        max_len = truncate*rate
        mfcc_feat = psf.mfcc(sig[:max_len], rate, nfft=nfft)
    else:
        mfcc_feat = psf.mfcc(sig, rate, nfft=nfft)
    if accel:
        return add_accel(np.asarray(mfcc_feat, dtype='float32'))
    else:
        return np.asarray(mfcc_feat, dtype='float32')

def delta(v, N=2, offset=1):
    d = np.zeros_like(v[:, offset:])
    for t in range(0, d.shape[0]):
        Z = 2 * sum(n**2 for n in range(1, N+1))
        d[t,:] = sum(n * (v[min(t+n, v.shape[0]-1), offset:]-v[max(t-n, 0), offset:]) for n in range(1,N+1)) / Z
    return d

def add_accel(row):
    return np.hstack([row, delta(row, N=2, offset=1), delta(delta(row, N=2, offset=1), offset=0)])


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

