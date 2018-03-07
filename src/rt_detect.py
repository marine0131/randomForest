#!/usr/bin/env python

from Data import sample
import os
import io
import numpy as np
import cPickle
import librosa
import soundfile as sf
import feature_extraction

import librosa.display
import matplotlib.pyplot as plt
from urllib2 import urlopen
from bottle import route, run, request


def generateDetectData(feat):
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the calculated angles are used as features.
    '''
    detectSamples = []

    for line in feat:
        features = np.array(line)
        detectSamples.append(sample(features))

    return detectSamples


def detect(rf, feat):
    '''
    detect a set of data
    '''
    # load random forest model
    with open(rf, 'rb') as f:
        testForest = cPickle.load(f)

    # generate feature array
    detectList = generateDetectData(feat)

    # classify sample or samples
    for samp in detectList:
        resultLabel = testForest.classify(samp)
        # print(resultLabel)

    return resultLabel


def plot(y, sr, point):
    plt.figure()
    plt.subplot()
    librosa.display.waveplot(y=y, sr=sr)
    plt.ylim(-1, 1)
    for i, p in enumerate(point):
        if i < len(point)-1:
            plt.plot([p[0], point[i+1][0]], [p[1], point[i+1][1]], 'r-')
    plt.show()
    plt.waitforbuttonpress()
    plt.close


def realtime_detect(af, af_url, rf):
    if not af:  # no local file, download from url
        # y, sr = sf.read(io.BytesIO(urlopen(af_url).read()))
        af = 'audio.mp3'
        with open(af, 'wb') as audio:
            audio.write(urlopen(af_url).read())
    y, sr = librosa.load(af)
    start = 0
    step = int(sr * 1)
    feat = []
    pos = 0
    total = 0
    feat = [feature_extraction.feature(y, sr)]
    result = detect(rf, feat)
    return result
    # while start+step < len(y):
    #     # print('slice from: {}'.format(start))
    #     slice_y = y[start: start+step]
    #     feat = [feature_extraction.feature(slice_y, sr)]
    #     resultLabel = detect(rf, feat)
    #     pos = pos+1 if resultLabel == 'positive' else pos
    #     total += 1
    #     start = start+step

    # if float(pos)/float(total) > 0.7:
    #     return 'positive'
    # return 'negative'


'''
arg1: random froest model
arg2: audio file
'''
# if __name__ == '__main__':
#     rf = os.path.join(os.path.abspath('..'), 'model/trainedForest_norm')
#     ff = sys.argv[1]
#
#     y, sr = librosa.load(ff)
#     start = 0
#     step = int(sr * 1)
#     feat = []
#     # print(y.shape)
#     point = []
#     while start+step < len(y):
#         # print('slice from: {}'.format(start))
#         slice_y = y[start: start+step]
#         feat = [feature_extraction.feature(slice_y, sr)]
#         resultLabel = detect(rf, feat)
#         # plot
#         result = 0.5 if resultLabel == 'positive' else 0.05
#         point.append([start/sr, result])
#         point.append([(start+step)/sr, result])
#
#         start = start+step
#
#     plot(y, sr, point)


'''
file: audio file absolute path in string
url: audio file download url
'''
@route('/')
def index():
    rf = os.path.join(os.path.abspath('..'), 'model/model_20180306')
    af = request.query.file
    af_url = request.query.url
    # ff = urllib2.urlopen(url)
    return realtime_detect(af, af_url, rf)

# run(host='192.168.10.12', port=2444)
rf = os.path.join(os.path.abspath('..'), 'model/model_20180306')
# af = '/home/whj/gitrepo/audioAnalysis/audio_file/positive'
af = "/home/whj/gitrepo/randomForest/audio_file/wKgA_FqeTsOADr38AABZa_M9MNE571.mp3"
af_url = None
print(realtime_detect(af, af_url, rf))
# for f in os.listdir(af):
#     ff = os.path.join(af, f)
#     print(realtime_detect(ff, af_url, rf))
