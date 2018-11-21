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
    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au) for au in audios]
    logging.info("Loading model")
    model = task.load(model_path)
    logging.info("Extracting convolutional states")
    conv_states = audiovis.conv_states(model, mfccs)
    logging.info("Extracting layer states")
    states = audiovis.layer_states(model, mfccs)
    logging.info("Extracting sentence embeddings")
    embeddings = audiovis.encode_sentences(model, mfccs)
    return {'mfcc': mfccs, 'conv_states': conv_states, 'layer_states': states, 'embeddings': embeddings}

def save_activations(audios, model_path, mfcc_path, conv_path, states_path, emb_path, accel=False):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au, format='mp3', accel=accel) for au in audios]
    logging.info("Loading model")
    model = task.load(model_path)
    audios = list(audios)
    numpy.save(mfcc_path, mfccs)
    logging.info("Extracting convolutional states")
    numpy.save(conv_path, audiovis.conv_states(model, mfccs))
    logging.info("Extracting layer states")
    numpy.save(states_path, audiovis.layer_states(model, mfccs))
    logging.info("Extracting sentence embeddings")
    numpy.save(emb_path, audiovis.encode_sentences(model, mfccs))


def save_flickr8k_val_activations():
    save_val_activations(dataset='flickr8k')

def save_places_val_activations():
    save_val_activations(dataset='places')


def save_val_activations(dataset='flickr8k'):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading data")
    model_path = "models/{}-speech/{}-speech.zip".format(dataset, dataset)
    if dataset == 'flickr8k':
        from imaginet.data_provider import getDataProvider
        prov = getDataProvider('flickr8k', root='.', audio_kind='human.max1K.accel3.ord.mfcc')
    elif dataset == 'places':
        from vg.places_provider import getDataProvider
        prov = getDataProvider('places', root='.', audio_kind='mfcc')

    mfcc = numpy.array([s['audio'] for s in prov.iterSentences(split='val') ])
    numpy.save("{}_val_mfcc.npy".format(dataset), mean(mfcc))
    text, spk = zip(*[ (s['raw'], s['speaker']) for s in prov.iterSentences(split='val') ])
    numpy.save("{}_val_text.npy".format(dataset), text)
    numpy.save("{}_val_spk.npy".format(dataset), spk)
    logging.info("Loading model")
    model = task.load(model_path)
    logging.info("Extracting convolutional states")
    numpy.save("{}_val_conv.npy".format(dataset), mean(audiovis.iter_conv_states(model, mfcc)))
    logging.info("Extracting layer states")
    numpy.save("{}_val_rec.npy".format(dataset), mean(audiovis.iter_layer_states(model, mfcc)))
    logging.info("Extracting utterance embeddings")
    numpy.save("{}_val_emb.npy".format(dataset), audiovis.encode_sentences(model, mfcc))

def mean(it):
    return numpy.array([ x.mean(axis=0) for x in it ])

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default=None,
                            help='Path to file with list of audio files to process, or sentences to synthesize.')
    parser.add_argument('--model', default="models/flickr8k-speech/flickr8k-speech.zip",
                            help='Path to file with model')
    parser.add_argument('--mfcc', default='mfcc.npy',
                            help='Path to file where MFCCs will be stored')
    parser.add_argument('--accel', action='store_true', default=False,
                            help='Extract delta and acceleration features in addition to plain MFCC features')
    parser.add_argument('--layer_states', default='states.npy',
                            help='Path to file where layer states will be stored')
    parser.add_argument('--conv_states', default='conv_states.npy',
                            help='Path to file where state of convolutional layer will be stored')
    parser.add_argument('--embeddings', default='embeddings.npy',
                            help='Path to file where sentence embeddings will be stored')
    parser.add_argument('--audio_dir', default="/tmp",
                            help='Path to directory where audio is stored')
    parser.add_argument('--synthesize', action='store_true', default=False,
                            help='Should audio be synthesized. If true `input` is assumed to be a file with sentences to synthesize.')

    args = parser.parse_args()
    if args.synthesize:
        texts = [ line.strip() for line in open(args.input)]
        audio.save_audio(texts, args.audio_dir)
        audio_paths = audio.audio_paths(texts, args.audio_dir)
    else:
        audio_paths = [ line.strip() for line in open(args.input)]
    save_activations(audio_paths, args.model,
        args.mfcc, args.conv_states, args.layer_states, args.embeddings, accel=args.accel)


if __name__=='__main__':
    main()
