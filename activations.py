import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
from vg.audio import extract_mfcc
import sys
import argparse
import logging
import vg.audio as audio


def activations(audios, model_path):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load(model_path)

    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au) for au in audios]
    logging.info("Extracting convolutional states")
    conv_states = audiovis.conv_states(model, mfccs)
    logging.info("Extracting layer states")
    states = audiovis.layer_states(model, mfccs)
    logging.info("Extracting sentence embeddings")
    embeddings = audiovis.encode_sentences(model, mfccs)
    return {'mfcc': mfccs, 'conv_states': conv_states, 'layer_states': states, 'embeddings': embeddings}

def save_activations(audios, model_path, mfcc_path, conv_path, states_path, emb_path):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load(model_path)
    audios = list(audios)
    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au) for au in audios]
    numpy.save(mfcc_path, mfccs)
    logging.info("Extracting convolutional states")
    numpy.save(conv_path, audiovis.conv_states(model, mfccs))
    logging.info("Extracting layer states")
    numpy.save(states_path, audiovis.layer_states(model, mfccs))
    logging.info("Extracting sentence embeddings")
    numpy.save(emb_path, audiovis.encode_sentences(model, mfccs))

def save_flickr8k_val_activations():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading data")
    model_path = "models/flickr8k-speech/flickr8k-speech.zip"
    from imaginet.data_provider import getDataProvider
    prov = getDataProvider('flickr8k', root='.', audio_kind='human.max1K.accel3.ord.mfcc')
    mfcc = numpy.array([s['audio'] for s in prov.iterSentences(split='val') ])
    numpy.save("flicr8k_val_mfcc.npy", mean(mfcc))
    text, spk = zip(*[ (s['raw'], s['speaker']) for s in prov.iterSentences(split='val') ])
    numpy.save("flickr8k_val_text.npy", text)
    numpy.save("flickr8k_val_spk.npy", spk)
    logging.info("Loading model")
    model = task.load(model_path)
    logging.info("Extracting convolutional states")
    numpy.save("flickr8k_val_conv.npy", mean(audiovis.iter_conv_states(model, mfcc)))
    logging.info("Extracting layer states")
    numpy.save("flickr8k_val_rec.npy", mean(audiovis.iter_layer_states(model, mfcc)))
    logging.info("Extracting utterance embeddings")
    numpy.save("flickr8k_val_emb.npy", audiovis.encode_sentences(model, mfcc))

def mean(it):
    return numpy.array([ x.mean(axis=0) for x in it ])

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument('texts', default=None,
                            help='Path to file with list of audio files')
    parser.add_argument('--model', default="models/flickr8k-speech/flickr8k-speech.zip",
                            help='Path to file with model')
    parser.add_argument('--mfcc', default='mfcc.npy',
                            help='Path to file where MFCCs will be stored')
    parser.add_argument('--layer_states', default='states.npy',
                            help='Path to file where layer states will be stored')
    parser.add_argument('--conv_states', default='conv_states.npy',
                            help='Path to file where state of convolutional layer will be stored')
    parser.add_argument('--embeddings', default='embeddings.npy',
                            help='Path to file where sentence embeddings will be stored')
    parser.add_argument('--audio_dir', default="/tmp",
                            help='Path to directory where audio is stored')
    parser.add_argument('--synthesize', action='store_true', default=False,
                            help='Should audio be synthesized')
    args = parser.parse_args()
    texts = [ line.strip() for line in open(args.texts)]
    if args.synthesize:
        audio.save_audio(texts, args.audio_dir)
    audios = audio.load_audio(texts, args.audio_dir)
    save_activations(audios, args.model,
        args.mfcc, args.conv_states, args.layer_states, args.embeddings)


if __name__=='__main__':
    main()
