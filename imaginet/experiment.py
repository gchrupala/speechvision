import numpy
import imaginet.simple_data as sd
import imaginet.task
from imaginet.evaluate import ranking, paraphrase_ranking
import json
from imaginet.simple_data import words
import random
from collections import Counter
import sys
import funktional.layer as layer
import os

def run_train(data, prov, model_config, run_config, eval_config, runid='', resume=False):
    epoch_evals = []
    if resume:
        last_epoch, path = last_dump(runid)
        print("Resuming from model {}".format(path))
        model = imaginet.task.load(path)
    else:
        last_epoch = 0
        print("Building model")
        model = imaginet.task.GenericBundle(dict(scaler=data.scaler,
                                             batcher=data.batcher), model_config, run_config['task'])

    print("Params {}".format(layer.param_count(model.params())))
    def epoch_eval(para=False):
        if para:
            return epoch_eval_para()
        task = model.task
        scaler = model.scaler
        batcher = model.batcher
        mapper = batcher.mapper
        sents = list(prov.iterSentences(split=eval_config['split']))
        sents_tok =  [ eval_config['tokenize'](sent) for sent in sents ]
        predictions = eval_config['encode_sentences'](model, sents_tok, batch_size=eval_config['batch_size'])
        images = list(prov.iterImages(split=eval_config['split']))
        img_fs = imaginet.task.encode_images(model, [ img['feat'] for img in images ])
        #img_fs = list(scaler.transform([ image['feat'] for image in images ]))
        correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                      for j in range(len(images)) ]
                                    for i in range(len(sents)) ] )
        result = ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)
        return result

    def epoch_eval_para():
        task = model.task
        scaler = model.scaler
        batcher = model.batcher
        mapper = batcher.mapper
        sents = list(prov.iterSentences(split=eval_config['split']))
        sents_tok =  [ eval_config['tokenize'](sent) for sent in sents ]
        predictions = eval_config['encode_sentences'](model, sents_tok, batch_size=eval_config['batch_size'])
        correct_para = numpy.array([ [ sents[i]['imgid']==sents[j]['imgid']
                                      for j in range(len(sents)) ]
                                    for i in range(len(sents)) ] )
        result = paraphrase_ranking(predictions, correct_para)
        return result


    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            result.append(model.task.loss_test(*model.task.args(item)))
        return result

    costs = Counter()
    for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        random.shuffle(data.data['train'])
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                cost = model.task.train(*model.task.args(item))
                costs += Counter({'cost':cost, 'N':1})
                print(epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])]))
                if j % run_config['validate_period'] == 0:
                        print(epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))]))
                sys.stdout.flush()

        model.save(path='model.r{}.e{}.zip'.format(runid,epoch))
        epoch_evals.append(epoch_eval(eval_config.get('para', False)))
        #print(epoch_evals[-1])
        #json.dump(epoch_evals[-1], open('scores.{}.json'.format(epoch),'w'))
        numpy.save('scores.{}.npy'.format(epoch), epoch_evals[-1])
    model.save(path='model.r{}.zip'.format(runid))
    return epoch_evals

def last_dump(runid):
    def epoch(name):
        return int(name.split(".")[2][1:])
    last = max([ epoch(f) for f in os.listdir(".") if f.startswith("model.r{}.".format(runid)) and f.endswith(".zip") ])
    return last, "model.r{}.e{}.zip".format(runid, last)

def run_eval(prov, config, encode_sentences=imaginet.task.encode_sentences, start_epoch=1, runid=''):
    datapath='/home/gchrupala/repos/reimaginet'
    for epoch in range(start_epoch, 1+config['epochs']):
        scores = evaluate(prov,
                          datapath=datapath,
                          tokenize=config['tokenize'],
                          split=config['split'],
                          task=config['task'],
                          encode_sentences=encode_sentences,
                          batch_size=config['batch_size'],
                          model_path='model.r{}.e{}.zip'.format(runid, epoch))
        json.dump(scores, open('scores.{}.json'.format(epoch),'w'))

def run_eval_para(prov, config, encode_sentences=imaginet.task.encode_sentences, start_epoch=1, runid=''):
    datapath='/home/gchrupala/repos/reimaginet'
    for epoch in range(start_epoch, 1+config['epochs']):
        scores = evaluate_para(prov,
                          datapath=datapath,
                          tokenize=config['tokenize'],
                          split=config['split'],
                          task=config['task'],
                          encode_sentences=encode_sentences,
                          batch_size=config['batch_size'],
                          model_path='model.r{}.e{}.zip'.format(runid, epoch))
        json.dump(scores, open('scores.{}.json'.format(epoch),'w'))



def evaluate(prov,
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             task=None,
             encode_sentences=imaginet.task.encode_sentences,
             tokenize=words,
             split='val'
            ):
    model = imaginet.task.load(path=model_path)
    task = model.task
    scaler = model.scaler
    batcher = model.batcher
    mapper = batcher.mapper
    sents = list(prov.iterSentences(split=split))
    sents_tok =  [ tokenize(sent) for sent in sents ]
    predictions = encode_sentences(model, sents_tok, batch_size=batch_size)
    images = list(prov.iterImages(split=split))
    img_fs = imaginet.task.encode_images(model, [ img['feat'] for img in images ])
    #img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sents)) ] )
    return ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)

def evaluate_para(prov,
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             task=None,
             encode_sentences=imaginet.task.encode_sentences,
             tokenize=words,
             split='val'
            ):
    model = imaginet.task.load(path=model_path)
    task = model.task
    scaler = model.scaler
    batcher = model.batcher
    mapper = batcher.mapper
    sents = list(prov.iterSentences(split=split))
    sents_tok =  [ tokenize(sent) for sent in sents ]
    predictions = encode_sentences(model, sents_tok, batch_size=batch_size)
    correct_para = numpy.array([ [ sents[i]['imgid']==sents[j]['imgid']
                                  for j in range(len(sents)) ]
                                for i in range(len(sents)) ] )
    return paraphrase_ranking(predictions, correct_para)
