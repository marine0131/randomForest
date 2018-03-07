#! /usr/bin/env python
import sys
import librosa
import numpy as np
# import time

DEBUG = False


def feature(y, sr=22050):
    feat = []

    # y, sr = librosa.load(f)
    # specgram(np.array(X), Fs=22050)
    # print("loaded {} data with {} hz".format(len(y), sr))

    # set the hop length, at 22050 hz, 512 samples ~= 23ms
    hop_length = 512

    # normalize
    y_norm = librosa.util.normalize(y)

    # time
    t = float(len(y))/float(sr)

    # average energy in second
    avg_energy = np.sum(y_norm**2) / t
    if DEBUG:
        print('avg_energy: {}'.format(avg_energy))

    # zero crossing
    # z = librosa.zero_crossings(y_norm)
    # z_num = len(z[z==True])
    # if DEBUG:
    #     print('zero crossing num: {}'.format(z_num))

    # zero-crossing rate
    z = librosa.feature.zero_crossing_rate(y_norm)
    z_mean = np.mean(z)
    if DEBUG:
        print('zero crossing rate: {}'.format(z.shape))

    feat.extend([avg_energy, z_mean])

    # compute stft and turn to db
    # D = librosa.amplitude_to_db(librosa.stft(norm_y), ref=np.max)
    if_gram, D = librosa.ifgram(y=y_norm, sr=sr, n_fft=2048, hop_length=hop_length)
    S, phase = librosa.magphase(D)

    # rms
    rms = librosa.feature.rmse(S=S)
    if DEBUG:
        print('rms shape: {}'.format(rms.shape))
    feat.append(np.mean(rms))

    # roll-off
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    if DEBUG:
        print('roll-off shape: {}'.format(rolloff.shape))
    feat.append(np.mean(rolloff))

    # centroid
    cent = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    if DEBUG:
        print('spectrum centroid shape: {}'.format(cent.shape))
    feat.append(np.mean(cent))

    # spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
    if DEBUG:
        print('spectral_bandwidth shape: {}'.format(spec_bw.shape))
    feat.append(np.mean(spec_bw))

    # tonnetz
    # y_harmonic = librosa.effects.harmonic(y_norm)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # if DEBUG:
    #     print('tonnetz shape: {}'.format(tonnetz.shape))
    # feat.extend(list(np.mean(tonnetz, axis=1)))

    # chroma cqt
    chroma_cq = librosa.feature.chroma_cqt(y=y_norm, sr=sr, n_chroma=12)
    if DEBUG:
        print('chroma cqt shape: {}'.format(chroma_cq.shape))
    feat.extend(list(np.mean(chroma_cq, axis=1)))

    # Chroma cens
    chroma_cens = librosa.feature.chroma_cens(y=y_norm, sr=sr, n_chroma=12)
    if DEBUG:
        print('chroma cens shape: {}'.format(chroma_cens.shape))
    feat.extend(list(np.mean(chroma_cens, axis=1)))

    # estimate global tempo
    oenv = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=hop_length)
    # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    # ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    # ac_global = librosa.util.normalize(ac_global)
    # estimate tempo
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    if DEBUG:
        print("tempo: {}".format(tempo))
    feat.append(tempo)

    # compute MFCC features from the raw signal
    # MEL = librosa.feature.melspectrogram(y=norm_y, sr=sr, hop_length=hop_length)

    return np.array(feat)


if __name__ == "__main__":
    f = sys.argv[1]
    feat = feature(f)
    print(feat)
