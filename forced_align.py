import json
import multiprocessing
import logging
import imaginet.defn.audiovis_rhn as audiovis
import numpy

ROOT = "/home/gchrupala/repos/reimaginet/data/synthetically-spoken-coco/mp3"
nthreads = multiprocessing.cpu_count()

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def align(audiopath, transcript):
    import gentle
    resources = gentle.Resources()
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                   conservative=False)
        return aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

def align_val(root=ROOT):
    coco = json.load(open("/home/gchrupala/repos/reimaginet/data/synthetically-spoken-coco/dataset.json"))
    log_level = "INFO"
    logging.getLogger().setLevel(log_level)
    with open("dataset.val.fa.json", 'w') as fa:
        for image in coco['images']:
            if image['split'] == 'val':
                for sent in image['sentences']:
                    audiopath = "{}/{}.mp3".format(root, sent['sentid'])
                    transcript = sent['raw']
                    result = json.loads(align(audiopath, transcript).to_json())
                    result['sentid'] = sent['sentid']
                    result['audiopath'] = audiopath
                    fa.write(json.dumps(result))
                    fa.write("\n")

def index(t):
    """Return index into the recurrent state of COCO model given timestep
    `t`.

    """
    return (t//10+6)//3

def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.

    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))

def slices(utt, rep, index=lambda ms: ms//10, aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.

    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))

def phoneme_states(alignments, reps, aggregate):
    """Return frames labeled with phonemes."""
    data_state =  [phoneme for (utt, rep) in zip(alignments, reps) for phoneme in slices(utt, rep) ]
    return data_state

def phoneme_train_data(alignment_path="data/coco/dataset.val.fa.json",
                  dataset_path="data/coco/dataset.json",
                  model_path="models/coco-speech.zip",
                  max_size=5000):
    """Generate data for training a phoneme decoding model."""
    import imaginet.data_provider as dp
    import imaginet.task as task

    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    data = {}
    for line in open(alignment_path):
        item = json.loads(line)
        data[item['sentid']] = item
    logging.info("Loading audio features")
    prov = dp.getDataProvider('coco', root='.', audio_kind='mfcc', load_img=False)
    val = list(prov.iterSentences(split='val'))
    logging.info("Loading RHN model")
    model_rhn = task.load(model_path)

    data_filter = [ (data[sent['sentid']], sent) for sent in val
                        if numpy.all([word.get('start', False) for word in data[sent['sentid']]['words']]) ]
    data_filter = data_filter[:max_size]
    
    logging.info("Extracting MFCC examples")
    data_state =  [phoneme for (utt, sent) in data_filter for phoneme in slices(utt, sent['audio']) ]
    save_fa_data(data_state, prefix="mfcc_")
    
    logging.info("Extracting convo states")
    states = audiovis.conv_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ])
    data_state =  [phoneme for i in range(len(data_filter))
                   for phoneme in slices(data_filter[i][0], states[i], index=index) ]
    save_fa_data(data_state, prefix="conv_")
    
    logging.info("Extracting recurrent layer states")
    states = audiovis.layer_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ], batch_size=32)
    for layer in range(0,5):
        def aggregate(x):
            return x[:,layer,:].mean(axis=0)
        data_state =  [phoneme for i in range(len(data_filter))
                       for phoneme in slices(data_filter[i][0], states[i], index=index, aggregate=aggregate) ]
        save_fa_data(data_state, prefix="rec{}_".format(layer))
        
def save_fa_data(data_state, prefix=""):
    y, X = zip(*data_state)
    X = numpy.vstack(X)
    y = numpy.array(y)
    numpy.save(prefix + "features.npy", X)
    numpy.save(prefix + "phonemes.npy", y)

    
