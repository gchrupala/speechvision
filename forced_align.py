import json
import multiprocessing
import logging
import numpy
import numpy as np
import pandas as pd

ROOT = "/home/gchrupala/repos/reimaginet/data/synthetically-spoken-coco/mp3"
nthreads = multiprocessing.cpu_count()

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))

try:
    import gentle
    resources = gentle.Resources()
except:
    pass

def align(audiopath, transcript):
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                   conservative=False)
        return aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

def align_val(root=ROOT, dataset_path="/home/gchrupala/repos/reimaginet/data/synthetically-spoken-coco/dataset.json"):
    data = json.load(open(dataset_path))
    log_level = "INFO"
    logging.getLogger().setLevel(log_level)
    with open("dataset.val.fa.json", 'w') as fa:
        for image in data['images']:
            if image['split'] == 'val':
                for sent in image['sentences']:
                    audiopath = "{}/{}.mp3".format(root, sent['sentid'])
                    transcript = sent['raw']
                    result = json.loads(align(audiopath, transcript).to_json())
                    result['sentid'] = sent['sentid']
                    result['audiopath'] = audiopath
                    fa.write(json.dumps(result))
                    fa.write("\n")

def index(t, stride=3):
    """Return index into the recurrent state of speech model given timestep
    `t`.

    """
    return (t//10+6)//stride

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
                  max_size=5000,
                  directory=".",
                  dataset='coco'):
    """Generate data for training a phoneme decoding model."""
    import imaginet.data_provider as dp
    import imaginet.task as task
    import imaginet.defn.audiovis_rhn as audiovis
    L = 5 if dataset == 'coco' else 4
    S = 3 if dataset == 'coco' else 2
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    data = {}
    for line in open(alignment_path):
        item = json.loads(line)
        data[item['sentid']] = item
    logging.info("Loading audio features")
    if dataset == 'coco':
        prov = dp.getDataProvider('coco', root='.', audio_kind='mfcc', load_img=False)
    elif dataset == 'flickr8k':
        prov = dp.getDataProvider('flickr8k', root='.', audio_kind='human.max1K.accel3.ord.mfcc', load_img=False)
    val = list(prov.iterSentences(split='val'))
    logging.info("Loading RHN model")
    model_rhn = task.load(model_path)

    data_filter = [ (data[sent['sentid']], sent) for sent in val
                        if numpy.all([word.get('start', False) for word in data[sent['sentid']]['words']]) ]
    data_filter = data_filter[:max_size]
    
    logging.info("Extracting MFCC examples")
    data_state =  [phoneme for (utt, sent) in data_filter for phoneme in slices(utt, sent['audio']) ]
    save_fa_data(data_state, prefix=directory + "/mfcc_")
    
    logging.info("Extracting convo states")
    states = audiovis.conv_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ])
    data_state =  [phoneme for i in range(len(data_filter))
                   for phoneme in slices(data_filter[i][0], states[i], index=lambda x: index(x, stride=S)) ]
    save_fa_data(data_state, prefix=directory + "/conv_")
    
    logging.info("Extracting recurrent layer states")
    states = audiovis.layer_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ], batch_size=32)

    for layer in range(0,L):
        def aggregate(x):
            return x[:,layer,:].mean(axis=0)
        data_state =  [phoneme for i in range(len(data_filter))
                       for phoneme in slices(data_filter[i][0], states[i], index=lambda x: index(x, stride=S), aggregate=aggregate) ]
        save_fa_data(data_state, prefix=directory + "/rec{}_".format(layer))
        
def save_fa_data(data_state, prefix=""):
    y, X = zip(*data_state)
    X = numpy.vstack(X)
    y = numpy.array(y)
    # Get rid of NaNs
    ix = np.isnan(X.sum(axis=1))
    X = X[~ix]
    y = y[~ix]
    numpy.save(prefix + "features.npy", X)
    numpy.save(prefix + "phonemes.npy", y)

def run_gentle(mp3_path):
    logging.basicConfig(level=logging.INFO)
    data = pd.read_csv(mp3_path + "/info.csv")
    fa_path = "{}/fa.json".format(mp3_path)
    fa = []    
    for i in range(data.shape[0]):
            transcript = data.iloc[i]['transcript']
            path = mp3_path + "/" + data.iloc[i]['path']
            logging.info("Processing file {}".format(path))
            result = json.loads(align(path, transcript).to_json())
            result['path'] =  data.iloc[i]['path']
            fa.append(result)
    json.dump(fa, open(fa_path, 'w'))
    
def phoneme_test_data(mp3_path="ganong", dataset='coco'): 
    """Generate data for ganong experiment"""
    rep_path = mp3_path + "-" + dataset
    L = 5 if dataset == 'coco' else 4
    S = 3 if dataset == 'coco' else 2
    try:
        fa_path = mp3_path + "/fa.json"
        fa = json.load(open(fa_path))
    except:
        run_gentle(mp3_path)
        fa = json.load(open(fa_path))
    FA = dict((utt['path'], utt) for utt in fa)
    data = pd.read_csv(mp3_path + "/info.csv")  
    alignment = [ FA[data.iloc[i]['path']] for i in range(data.shape[0]) ]
    logging.info("Extracting MFCC examples")
    states = np.load(rep_path + "/mfcc.npy", encoding='bytes')
    save_fa_data([phoneme for (utt, state) in zip(alignment, states) for phoneme in slices(utt, state) ], 
                  prefix=rep_path + "/test_mfcc_")
    logging.info("Extracting convolutional examples")
    states = np.load(rep_path + "/conv_states.npy", encoding='bytes')
    save_fa_data([phoneme for (utt, state) in zip(alignment, states) for phoneme in slices(utt, state, index=lambda x: index(x, stride=S)) ], 
                  prefix=rep_path + "/test_conv_")
    logging.info("Extracting recurrent examples")
    states = np.load(rep_path + "/states.npy", encoding='bytes')
    for layer in range(0, L):
        def aggregate(x):
            return x[:,layer,:].mean(axis=0)
        save_fa_data([phoneme for (utt, state) in zip(alignment, states) 
                      for phoneme in slices(utt, state, index=lambda x: index(x, stride=S), aggregate=aggregate) ],
                      prefix=rep_path + "/test_rec{}_".format(layer))

def train_test_split(X, y):
    I = (X.shape[0]//3)*2
    X_train = X[:I, :]
    y_train = y[:I]
    X_val = X[I:,:]
    y_val = y[I:]
    return X_train, X_val, y_train, y_val

def decoding(rep, directory=".", max_size=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='lbfgs', random_state=123)     
    X = np.load(directory  + "/" + rep + "_features.npy")
    y = np.load(directory  + "/" + rep + "_phonemes.npy")
    if max_size is not None:
        X = X[:max_size,:]
        y = y[:max_size]
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model.fit(X_train, y_train)
    X_val = scaler.transform(X_val)
    val_score = 1-model.score(X_val, y_val)
    W = np.load(directory  + "/" + "test_" + rep + "_features.npy")
    W = scaler.transform(W)
    z = np.load(directory  + "/" + "test_" + rep + "_phonemes.npy")
    test_score = 1-model.score(W, z)
    return (val_score, test_score)


def run_decoding(mp3_path, model_type='coco'):
    import os
    import os.path
    import activations as A
    data = pd.read_csv(mp3_path + "/info.csv")
    audio_paths = [ mp3_path + "/" + path for  path in data['path'].as_matrix() ]
    model = "models/{}-speech.zip".format(model_type)
    directory = "{}-{}".format(mp3_path, model_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    A.save_activations(audio_paths, model, 
                       directory + "/mfcc.npy", 
                       directory + "/conv_states.npy", 
                       directory + "/states.npy",
                       directory + "/embeddings.npy", 
                       accel=model_type == 'flickr8k', 
                       nfft=1024)    
    phoneme_train_data(alignment_path="data/{}/dataset.val.fa.json".format(model_type),
                      dataset_path="data/{}/dataset.json".format(model_type),
                      model_path=model,
                      max_size=5000,
                      directory=directory,
                      dataset=model_type)
    phoneme_test_data(mp3_path=mp3_path, dataset=model_type)
    if model_type == 'coco':
        reps = ["mfcc", "conv", "rec0", "rec1", "rec2", "rec3", "rec4"]
    elif model_type == 'flickr8k':
        reps = ["mfcc", "conv", "rec0", "rec1", "rec2", "rec3"]
    print("rep val test")
    for rep in reps:
        val, test = decoding(rep, directory=directory, max_size=50000)
        print("{} {:.3} {:.3}".format(rep, val, test))     

