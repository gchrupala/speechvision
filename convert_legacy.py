import json
import numpy
import io
import zipfile
import pickle
import sys
from imaginet.task import GenericBundle


def load(path):
    """Load data and reconstruct model."""
    with zipfile.ZipFile(path,'r') as zf:
        buf = io.BytesIO(zf.read('weights.npy'))
        weights = numpy.load(buf, encoding='bytes')
        config  = json.loads(zf.read('config.json').decode('utf-8'))
        data  = pickle.loads(zf.read('data.pkl'))
        task = pickle.loads(bytes(config['task'], 'utf-8'))
    return GenericBundle(data, config, task, weights=weights)


def main():
    path_in = sys.argv[1]
    path_out = sys.argv[2]
    model = load(path_in)
    model.save(path_out)
    

main()

