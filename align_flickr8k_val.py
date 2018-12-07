import forced_align as A
import json
import logging

   
def align_val():
    wav2id = {}
    id2wav = {}
    for line in open("data/flickr8k/flickr_audio/wav2capt.txt"):
        wav, jpg, i = line.split()
        wav2id[wav] = jpg+i
        id2wav[jpg+i] = wav
    id2capt = {}
    for line in open("data/flickr8k/Flickr8k.token.txt"):
        xs = line.split()
        id2capt[xs[0]] = " ".join(xs[1:])
    data = json.load(open("data/flickr8k/dataset.json"))
    log_level = "INFO"
    logging.getLogger().setLevel(log_level)
    with open("data/flickr8k/dataset.val.fa.json", 'w') as fa:
        for image in data['images']:
            if image['split'] == 'val':
                for i, sent in enumerate(image['sentences']):
                    audiopath = "data/flickr8k/flickr_audio/wavs/{}".format(id2wav["{}#{}".format(image['filename'], i)])
                    transcript = sent['raw']
                    result = json.loads(A.align(audiopath, transcript).to_json())
                    result['sentid'] = sent['sentid']
                    result['audiopath'] = audiopath
                    fa.write(json.dumps(result))
                    fa.write("\n")


