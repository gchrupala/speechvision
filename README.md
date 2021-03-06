# speechvision



Flickr8k Data, validation 
=========================

All files below contain utterance data in the same order.
The data and models can be downloaded from: https://drive.google.com/open?id=1junt1_4Rk-Xdw8omz6MxxPp4tYN-axCZ


- `flickr8k_val_text.npy`: text of each utterance read by crowd workers
- `flickr8k_val_spk.npy`: speaker ID for each utterances
- `flicr8k_val_mfcc.npy`:  mean MFCC features for each utterance
- `flickr8k_val_conv.npy`: mean convolutional layer activations
- `flickr8k_val_rec.npy`:  mean recurrent layer activations (for each of 4 layers)
- `flickr8k_val_emb.npy`:  utterance embeddings (after the self-attention layer)

The mean activations (average over time) were extracted from model flickr8k-speech.zip, trained as described in:

Chrupała, G., Gelderloos, L., & Alishahi, A. (2017). Representations of language in a model of visually grounded speech signal. ACL. arXiv preprint: https://arxiv.org/abs/1702.01991


Places Data, validation
=======================

All files below contain utterance data in the same order.
The data and models can be downloaded from: https://drive.google.com/file/d/17OSXke01YsgzyCo6dZ-QNfLKFPxyGxom/view?usp=sharing



- `places_val_text.npy`: text of each utterance read by crowd workers
- `places_val_spk.npy`: speaker ID for each utterances
- `places_val_mfcc.npy`:  mean MFCC features for each utterance
- `places_val_conv.npy`: mean convolutional layer activations
- `places_val_rec.npy`:  mean recurrent layer activations (for each of 4 layers)
- `places_val_emb.npy`:  utterance embeddings (after the self-attention layer)

The mean activations (average over time) were extracted from model ...
