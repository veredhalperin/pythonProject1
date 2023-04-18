from CPC18PsychForestPython.CPC18_getDist import CPC18_getDist
import warnings
import numpy as np
import copy
warnings.filterwarnings("ignore")
from datetime import datetime
import logging
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.autograd.set_detect_anomaly(True)
biases = ['unb', 'uni', 'pes', 'sig']


def st_probs(val_dist, Corr, probs, differences, probs0, differences0):
    """
    :param val_dist: values and distribution of each sampling tool on gambles a and b
    :param Corr: correlation between gambles a and b,
    :param probs:the probabilities for all of the  possible delta ST for for trials 1-4,
    :param differences: all of the possible delta ST for trials 1-4,
    :param probs0:the probabilities for all of the  possible delta ST for trial 0,
    :param differences0:all of the possible delta ST for trial 0,

    this function returns all of the possible delta ST and their probabilities
    """
    biases = ['unb', 'uni', 'pes', 'sig']
    kapa, st_difference = {}, {}
    for i in range(0, 4):
        kapa[biases[i]] = [], []
        st_difference[biases[i]] = [], []
        for j in range(0, 4):
            kapa[biases[i] + '_' + biases[j]] = [], []
            st_difference[biases[i] + '_' + biases[j]] = [], []
            for k in range(0, 4):
                kapa[biases[i] + '_' + biases[j] + '_' + biases[k]] = [], []
                st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]] = [], []
    for i in range(0, 4):
        val_A1, dist_A1, val_B1, dist_B1 = val_dist[biases[i]][0], val_dist[biases[i]][1], val_dist[biases[i]][2], \
                                           val_dist[biases[i]][3]
        for i1 in range(val_A1.shape[0]):
            for j1 in range(val_B1.shape[0]):
                if min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]):
                    kapa[biases[i]][0].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1]))
                    st_difference[biases[i]][0].append(val_B1[j1] - val_A1[i1])
                    for j in range(0, 4):
                        val_A2, dist_A2, val_B2, dist_B2 = val_dist[biases[j]][0], val_dist[biases[j]][1], \
                                                           val_dist[biases[j]][2], val_dist[biases[j]][3]
                        for i2 in range(val_A2.shape[0]):
                            for j2 in range(val_B2.shape[0]):
                                if min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]):
                                    kapa[biases[i] + '_' + biases[j]][0].append(
                                        (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (
                                                    min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],
                                                                                                dist_B2[j2])))
                                    st_difference[biases[i] + '_' + biases[j]][0].append(
                                        (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                    for k in range(0, 4):
                                        val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], val_dist[biases[k]][
                                            1], val_dist[biases[k]][2], val_dist[biases[k]][3]
                                        for i3 in range(val_A3.shape[0]):
                                            for j3 in range(val_B3.shape[0]):
                                                if min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                               dist_B3[j3]):
                                                    kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][0].append((
                                                                                                                              min(
                                                                                                                                  dist_A1[
                                                                                                                                      i1 + 1],
                                                                                                                                  dist_B1[
                                                                                                                                      j1 + 1]) - max(
                                                                                                                          dist_A1[
                                                                                                                              i1],
                                                                                                                          dist_B1[
                                                                                                                              j1])) * (
                                                                                                                              min(
                                                                                                                                  dist_A2[
                                                                                                                                      i2 + 1],
                                                                                                                                  dist_B2[
                                                                                                                                      j2 + 1]) - max(
                                                                                                                          dist_A2[
                                                                                                                              i2],
                                                                                                                          dist_B2[
                                                                                                                              j2])) * (
                                                                                                                              min(
                                                                                                                                  dist_A3[
                                                                                                                                      i3 + 1],
                                                                                                                                  dist_B3[
                                                                                                                                      j3 + 1]) - max(
                                                                                                                          dist_A3[
                                                                                                                              i3],
                                                                                                                          dist_B3[
                                                                                                                              j3])))
                                                    st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                        0].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                if Corr == -1:
                    if min(dist_A1[i1 + 1], 1 - dist_B1[j1]) > max(dist_A1[i1], 1 - dist_B1[j1 + 1]) and biases[i] in (
                    'unb', 'sig'):
                        kapa[biases[i]][1].append(
                            min(dist_A1[i1 + 1], 1 - dist_B1[j1]) - max(dist_A1[i1], 1 - dist_B1[j1 + 1]))
                        st_difference[biases[i]][1].append(val_B1[j1] - val_A1[i1])
                        for j in range(0, 4):
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[biases[j]][0], val_dist[biases[j]][1], \
                                                               val_dist[biases[j]][2], val_dist[biases[j]][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2], 1 - dist_B2[j2 + 1]) and \
                                            biases[j] in ('unb', 'sig'):
                                        kapa[biases[i] + '_' + biases[j]][1].append((min(dist_A1[i1 + 1],
                                                                                         1 - dist_B1[j1]) - max(
                                            dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],
                                                                                      1 - dist_B2[j2]) - max(
                                            dist_A2[i2], 1 - dist_B2[j2 + 1])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],
                                                                                                   1 - dist_B3[
                                                                                                       j3 + 1]) and \
                                                            biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B1[
                                                                                                                                          j1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              1 -
                                                                                                                              dist_B1[
                                                                                                                                  j1 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B2[
                                                                                                                                          j2]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              1 -
                                                                                                                              dist_B2[
                                                                                                                                  j2 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B3[
                                                                                                                                          j3]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              1 -
                                                                                                                              dist_B3[
                                                                                                                                  j3 + 1])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B1[
                                                                                                                                          j1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              1 -
                                                                                                                              dist_B1[
                                                                                                                                  j1 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B2[
                                                                                                                                          j2]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              1 -
                                                                                                                              dist_B2[
                                                                                                                                  j2 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and \
                                            biases[j] in ('uni', 'pes'):
                                        kapa[biases[i] + '_' + biases[j]][1].append((min(dist_A1[i1 + 1],
                                                                                         1 - dist_B1[j1]) - max(
                                            dist_A1[i1], 1 - dist_B1[j1 + 1])) * (min(dist_A2[i2 + 1],
                                                                                      dist_B2[j2 + 1]) - max(
                                            dist_A2[i2], dist_B2[j2])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],
                                                                                                   1 - dist_B3[
                                                                                                       j3 + 1]) and \
                                                            biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B1[
                                                                                                                                          j1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              1 -
                                                                                                                              dist_B1[
                                                                                                                                  j1 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B3[
                                                                                                                                          j3]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              1 -
                                                                                                                              dist_B3[
                                                                                                                                  j3 + 1])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B1[
                                                                                                                                          j1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              1 -
                                                                                                                              dist_B1[
                                                                                                                                  j1 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and biases[i] in (
                    'uni', 'pes'):
                        kapa[biases[i]][1].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1]))
                        st_difference[biases[i]][1].append(val_B1[j1] - val_A1[i1])
                        for j in range(0, 4):
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[biases[j]][0], val_dist[biases[j]][1], \
                                                               val_dist[biases[j]][2], val_dist[biases[j]][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if min(dist_A2[i2 + 1], 1 - dist_B2[j2]) > max(dist_A2[i2], 1 - dist_B2[j2 + 1]) and \
                                            biases[j] in ('unb', 'sig'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (
                                                        min(dist_A2[i2 + 1], 1 - dist_B2[j2]) - max(dist_A2[i2],
                                                                                                    1 - dist_B2[
                                                                                                        j2 + 1])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],
                                                                                                   1 - dist_B3[
                                                                                                       j3 + 1]) and \
                                                            biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B2[
                                                                                                                                          j2]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              1 -
                                                                                                                              dist_B2[
                                                                                                                                  j2 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B3[
                                                                                                                                          j3]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              1 -
                                                                                                                              dist_B3[
                                                                                                                                  j3 + 1])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B2[
                                                                                                                                          j2]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              1 -
                                                                                                                              dist_B2[
                                                                                                                                  j2 + 1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and \
                                            biases[j] in ('uni', 'pes'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (
                                                        min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],
                                                                                                    dist_B2[j2])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if min(dist_A3[i3 + 1], 1 - dist_B3[j3]) > max(dist_A3[i3],
                                                                                                   1 - dist_B3[
                                                                                                       j3 + 1]) and \
                                                            biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      1 -
                                                                                                                                      dist_B3[
                                                                                                                                          j3]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              1 -
                                                                                                                              dist_B3[
                                                                                                                                  j3 + 1])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                elif Corr == 0:
                    if biases[i] in ('unb', 'sig'):
                        kapa[biases[i]][1].append((dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]))
                        st_difference[biases[i]][1].append(val_B1[j1] - val_A1[i1])
                        for j in range(0, 4):
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[biases[j]][0], val_dist[biases[j]][1], \
                                                               val_dist[biases[j]][2], val_dist[biases[j]][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if biases[j] in ('unb', 'sig'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                        dist_A2[i2 + 1] - dist_A2[i2]) * (
                                                        dist_B2[j2 + 1] - dist_B2[j2]))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append(
                                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (
                                                                        dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                                        dist_A2[i2 + 1] - dist_A2[i2]) * (
                                                                        dist_B2[j2 + 1] - dist_B2[j2]) * (
                                                                        dist_A3[i3 + 1] - dist_A3[i3]) * (
                                                                        dist_B3[j3 + 1] - dist_B3[j3]))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append(
                                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (
                                                                        dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                                        dist_A2[i2 + 1] - dist_A2[i2]) * (
                                                                        dist_B2[j2 + 1] - dist_B2[j2]) * (
                                                                        min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(
                                                                    dist_A3[i3], dist_B3[j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and \
                                            biases[j] in ('uni', 'pes'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                        min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],
                                                                                                    dist_B2[j2])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append(
                                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (
                                                                        dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                                        min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(
                                                                    dist_A2[i2], dist_B2[j2])) * (
                                                                        dist_A3[i3 + 1] - dist_A3[i3]) * (
                                                                        dist_B3[j3 + 1] - dist_B3[j3]))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append(
                                                            (dist_A1[i1 + 1] - dist_A1[i1]) * (
                                                                        dist_B1[j1 + 1] - dist_B1[j1]) * (
                                                                        min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(
                                                                    dist_A2[i2], dist_B2[j2])) * (
                                                                        min(dist_A3[i3 + 1], dist_B3[j3 + 1]) - max(
                                                                    dist_A3[i3], dist_B3[j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                    elif min(dist_A1[i1 + 1], dist_B1[j1 + 1]) > max(dist_A1[i1], dist_B1[j1]) and biases[i] in (
                    'uni', 'pes'):
                        kapa[biases[i]][1].append(min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1]))
                        st_difference[biases[i]][1].append(val_B1[j1] - val_A1[i1])
                        for j in range(0, 4):
                            val_A2, dist_A2, val_B2, dist_B2 = val_dist[biases[j]][0], val_dist[biases[j]][1], \
                                                               val_dist[biases[j]][2], val_dist[biases[j]][3]
                            for i2 in range(val_A2.shape[0]):
                                for j2 in range(val_B2.shape[0]):
                                    if biases[j] in ('unb', 'sig'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (
                                                        dist_A2[i2 + 1] - dist_A2[i2]) * (
                                                        dist_B2[j2 + 1] - dist_B2[j2]))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  dist_A2[
                                                                                                                                      i2 + 1] -
                                                                                                                                  dist_A2[
                                                                                                                                      i2]) * (
                                                                                                                                  dist_B2[
                                                                                                                                      j2 + 1] -
                                                                                                                                  dist_B2[
                                                                                                                                      j2]) * (
                                                                                                                                  dist_A3[
                                                                                                                                      i3 + 1] -
                                                                                                                                  dist_A3[
                                                                                                                                      i3]) * (
                                                                                                                                  dist_B3[
                                                                                                                                      j3 + 1] -
                                                                                                                                  dist_B3[
                                                                                                                                      j3]))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  dist_A2[
                                                                                                                                      i2 + 1] -
                                                                                                                                  dist_A2[
                                                                                                                                      i2]) * (
                                                                                                                                  dist_B2[
                                                                                                                                      j2 + 1] -
                                                                                                                                  dist_B2[
                                                                                                                                      j2]) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                    elif min(dist_A2[i2 + 1], dist_B2[j2 + 1]) > max(dist_A2[i2], dist_B2[j2]) and \
                                            biases[j] in ('uni', 'pes'):
                                        kapa[biases[i] + '_' + biases[j]][1].append(
                                            (min(dist_A1[i1 + 1], dist_B1[j1 + 1]) - max(dist_A1[i1], dist_B1[j1])) * (
                                                        min(dist_A2[i2 + 1], dist_B2[j2 + 1]) - max(dist_A2[i2],
                                                                                                    dist_B2[j2])))
                                        st_difference[biases[i] + '_' + biases[j]][1].append(
                                            (val_B1[j1] + val_B2[j2]) / 2 - (val_A1[i1] + val_A2[i2]) / 2)
                                        for k in range(0, 4):
                                            val_A3, dist_A3, val_B3, dist_B3 = val_dist[biases[k]][0], \
                                                                               val_dist[biases[k]][1], \
                                                                               val_dist[biases[k]][2], \
                                                                               val_dist[biases[k]][3]
                                            for i3 in range(val_A3.shape[0]):
                                                for j3 in range(val_B3.shape[0]):
                                                    if biases[k] in ('unb', 'sig'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  dist_A3[
                                                                                                                                      i3 + 1] -
                                                                                                                                  dist_A3[
                                                                                                                                      i3]) * (
                                                                                                                                  dist_B3[
                                                                                                                                      j3 + 1] -
                                                                                                                                  dist_B3[
                                                                                                                                      j3]))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
                                                    elif min(dist_A3[i3 + 1], dist_B3[j3 + 1]) > max(dist_A3[i3],
                                                                                                     dist_B3[j3]) and \
                                                            biases[k] in ('uni', 'pes'):
                                                        kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1].append((
                                                                                                                                  min(
                                                                                                                                      dist_A1[
                                                                                                                                          i1 + 1],
                                                                                                                                      dist_B1[
                                                                                                                                          j1 + 1]) - max(
                                                                                                                              dist_A1[
                                                                                                                                  i1],
                                                                                                                              dist_B1[
                                                                                                                                  j1])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A2[
                                                                                                                                          i2 + 1],
                                                                                                                                      dist_B2[
                                                                                                                                          j2 + 1]) - max(
                                                                                                                              dist_A2[
                                                                                                                                  i2],
                                                                                                                              dist_B2[
                                                                                                                                  j2])) * (
                                                                                                                                  min(
                                                                                                                                      dist_A3[
                                                                                                                                          i3 + 1],
                                                                                                                                      dist_B3[
                                                                                                                                          j3 + 1]) - max(
                                                                                                                              dist_A3[
                                                                                                                                  i3],
                                                                                                                              dist_B3[
                                                                                                                                  j3])))
                                                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][
                                                            1].append((val_B1[j1] + val_B2[j2] + val_B3[j3]) / 3 - (
                                                                    val_A1[i1] + val_A2[i2] + val_A3[i3]) / 3)
    for i in range(0, 4):
        probs0[biases[i]].append(np.array(kapa[biases[i]][0]) / 3)
        differences0[biases[i]].append(np.array(st_difference[biases[i]][0]))
        for j in range(0, 4):
            probs0[biases[i] + '_' + biases[j]].append(np.array(kapa[biases[i] + '_' + biases[j]][0]) / 3)
            differences0[biases[i] + '_' + biases[j]].append(np.array(st_difference[biases[i] + '_' + biases[j]][0]))
            for k in range(0, 4):
                probs0[biases[i] + '_' + biases[j] + '_' + biases[k]].append(
                    np.array(kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][0]) / 3)
                differences0[biases[i] + '_' + biases[j] + '_' + biases[k]].append(
                    np.array(st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][0]))
    if Corr != 1:
        for i in range(0, 4):
            probs[biases[i]].append(np.array(kapa[biases[i]][1]) / 3)
            differences[biases[i]].append(np.array(st_difference[biases[i]][1]))
            for j in range(0, 4):
                probs[biases[i] + '_' + biases[j]].append(np.array(kapa[biases[i] + '_' + biases[j]][1]) / 3)
                differences[biases[i] + '_' + biases[j]].append(np.array(st_difference[biases[i] + '_' + biases[j]][1]))
                for k in range(0, 4):
                    probs[biases[i] + '_' + biases[j] + '_' + biases[k]].append(
                        np.array(kapa[biases[i] + '_' + biases[j] + '_' + biases[k]][1]) / 3)
                    differences[biases[i] + '_' + biases[j] + '_' + biases[k]].append(np.array(
                        st_difference[biases[i] + '_' + biases[j] + '_' + biases[k]][1]))
    else:
        probs = copy.deepcopy(probs0)
        differences = copy.deepcopy(differences0)
    return probs, differences, probs0, differences0


def preprocess(csvName, pklName):
    """
    :param csvName: name of the csv file it based on, without '.csv' ending
    it needs to be in this shape (TEST_TRAIN can be 'TEST' or 'TRAIN', and Problem is the GameID):
    Ha	pHa	La	Hb	pHb	Lb	LotShapeA	LotNumA	LotShapeB	LotNumB	B_rate	TEST_TRAIN	Problem
    :param pklName: name of the pickle files the function will create (without the numbers or .pkl at at the end)

    this function creates pickle files (a file for each 100 games) with the weights+ST possible values vectors of all ST combinations per game
    each 100 games has _number ending- first 100 games are ended with _100 etc.
    this funtion pre-process the data before the network, you run this only once!!
    """
    Data = pd.read_csv(csvName + '.csv')
    BEVas, BEVbs = [], []
    probs, differences, probs0, differences0 = {}, {}, {}, {}
    for i in range(0,4):
        probs[biases[i]], differences[biases[i]], probs0[biases[i]], differences0[biases[i]] = [], [], [], []
        for j in range(0,4):
            probs[biases[i] + '_' + biases[j]], differences[biases[i] + '_' + biases[j]], probs0[biases[i] + '_' + biases[j]], differences0[
                biases[i] + '_' + biases[j]] = [], [], [], []
            for k in range(0,4):
                probs[biases[i] + '_' + biases[j] + '_' + biases[k]], differences[biases[i] + '_' + biases[j] + '_' + biases[k]], probs0[
                    biases[i] + '_' + biases[j] + '_' + biases[k]], differences0[biases[i] + '_' + biases[j] + '_' + biases[k]] = [], [], [], []
    nProblems = Data.shape[0]
    Data.index = range(nProblems)
    for prob in range(nProblems):
        Ha = Data['Ha'][prob]
        pHa = Data['pHa'][prob]
        La = Data['La'][prob]
        LotShapeA = Data['LotShapeA'][prob]
        LotNumA = int(Data['LotNumA'][prob])
        Hb = Data['Hb'][prob]
        pHb = Data['pHb'][prob]
        Lb = Data['Lb'][prob]
        LotShapeB = Data['LotShapeB'][prob]
        LotNumB = int(Data['LotNumB'][prob])
        Corr = Data['Corr'][prob]
        DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
        DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
        nA = DistA.shape[0]
        nB = DistB.shape[0]
        MinA = DistA[0, 0]
        MinB = DistB[0, 0]
        MaxOutcome = np.maximum(DistA[nA - 1, 0], DistB[nB - 1, 0])
        SignMax = np.sign(MaxOutcome)
        if MinA == MinB:
            RatioMin = 1
        elif np.sign(MinA) == np.sign(MinB):
            RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
        else:
            RatioMin = 0
        Range = MaxOutcome - min(MinA, MinB)
        BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
        BEVb = np.matrix.dot(DistB[:, 0], DistB[:, 1])
        BEVas.append(BEVa)
        BEVbs.append(BEVb)
        val_dist = {}
        val_dist['unb'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(DistA[:, 1]))), DistB[:, 0],
                           np.concatenate(([0], np.cumsum(DistB[:, 1])))]
        val_dist['uni'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(np.repeat([1 / nA], nA)))), DistB[:, 0],
                           np.concatenate(([0], np.cumsum(np.repeat([1 / nB], nB))))]
        if SignMax > 0 and RatioMin < 0.25:
            val_dist['pes'] = [np.array([MinA]), np.concatenate(([0], [1])), np.array([MinB]),
                               np.concatenate(([0], [1]))]
        else:
            val_dist['pes'] = [DistA[:, 0], np.concatenate(([0], np.cumsum(np.repeat([1 / nA], nA)))), DistB[:, 0],
                               np.concatenate(([0], np.cumsum(np.repeat([1 / nB], nB))))]
        val_dist['sig'] = [Range * np.sign(DistA[:, 0]), np.concatenate(([0], np.cumsum(DistA[:, 1]))),
                           Range * np.sign(DistB[:, 0]), np.concatenate(([0], np.cumsum(DistB[:, 1])))]
        probs, differences, probs0, differences0 = st_probs(val_dist, Corr, probs, differences, probs0,
                                                                   differences0)
        if (prob + 1) % 100 == 0 or prob == nProblems - 1:
            if (prob + 1) % 100 == 0:
                tmp = Data[prob - 99:prob + 1]
            elif prob == nProblems - 1:
                tmp = Data[prob - prob % 100:]
            tmp.index = range(len(tmp))
            tmp['BEVa'] = BEVas
            tmp['BEVb'] = BEVbs
            tmp = tmp[['Problem', 'BEVa', 'BEVb']]
            for key in probs.keys():
                tmp['Weight_' + key] = probs[key]
            for key in differences.keys():
                tmp['ST_' + key] = differences[key]
            for key in probs0.keys():
                tmp['Weight0_' + key] = probs0[key]
            for key in differences0.keys():
                tmp['ST0_' + key] = differences0[key]
            tmp.to_pickle(pklName + '_' + str(prob + 1) + '.pkl')
            BEVas, BEVbs = [], []
            probs, differences, probs0, differences0 = {}, {}, {}, {}
            for i in range(0, 4):
                probs[biases[i]], differences[biases[i]], probs0[biases[i]], differences0[biases[i]] = [], [], [], []
                for j in range(0, 4):
                    probs[biases[i] + '_' + biases[j]], differences[biases[i] + '_' + biases[j]], probs0[
                        biases[i] + '_' + biases[j]], differences0[
                        biases[i] + '_' + biases[j]] = [], [], [], []
                    for k in range(0, 4):
                        probs[biases[i] + '_' + biases[j] + '_' + biases[k]], differences[
                            biases[i] + '_' + biases[j] + '_' + biases[k]], probs0[
                            biases[i] + '_' + biases[j] + '_' + biases[k]], differences0[
                            biases[i] + '_' + biases[j] + '_' + biases[k]] = [], [], [], []


def Sigmoid(x, B):
    return 1 / (1 + torch.exp(-1 * B * x))


def calc_probs(probs2):
    """
    :param probs2: sampling tool probabilities

    this function returns the probability for each sampling tool combination
    """
    probs = torch.tensor([], dtype=torch.float64, requires_grad=True)
    for prob0 in probs2:
        probs = torch.cat((probs, torch.reshape(prob0, (1,))))
        for prob1 in probs2:
            probs = torch.cat((probs, torch.reshape(prob0 * prob1, (1,))))
            for prob2 in probs2:
                probs = torch.cat((probs, torch.reshape(prob0 * prob1 * prob2, (1,))))
    return torch.reshape(probs, (1, probs.shape[0]))


class BEASTNet(torch.nn.Module):
    def __init__(self, probs0, probs1, probs2, probs3, probs4, BEV, b):
        super(BEASTNet, self).__init__()
        self.probs0 = torch.nn.Parameter(torch.tensor(probs0, dtype=torch.float64))
        self.probs1 = torch.nn.Parameter(torch.tensor(probs1, dtype=torch.float64))
        self.probs2 = torch.nn.Parameter(torch.tensor(probs2, dtype=torch.float64))
        self.probs3 = torch.nn.Parameter(torch.tensor(probs3, dtype=torch.float64))
        self.probs4 = torch.nn.Parameter(torch.tensor(probs4, dtype=torch.float64))
        self.BEV = torch.nn.Parameter(torch.tensor(BEV, dtype=torch.float64))
        self.B = torch.nn.Parameter(torch.tensor(b, dtype=torch.float64))

    def forward(self, BEV, ST0, Weight0, ST1, Weight1):
        probs0 = calc_probs(torch.nn.functional.softmax(self.probs0))
        probs1 = calc_probs(torch.nn.functional.softmax(self.probs1))
        probs2 = calc_probs(torch.nn.functional.softmax(self.probs2))
        probs3 = calc_probs(torch.nn.functional.softmax(self.probs3))
        probs4 = calc_probs(torch.nn.functional.softmax(self.probs4))
        tmp0 = torch.tensor([], dtype=torch.float64, requires_grad=True)
        tmp1 = torch.tensor([], dtype=torch.float64, requires_grad=True)
        for i in range(len(Weight0)):
            tmp0 = torch.cat((tmp0, torch.matmul(
                1 / (1 + torch.exp(-self.B * (torch.nn.functional.relu(self.BEV) * BEV + ST0[i]))), Weight0[i])))
        for i in range(len(Weight1)):
            tmp1 = torch.cat((tmp1, torch.matmul(
                1 / (1 + torch.exp(-self.B * (torch.nn.functional.relu(self.BEV) * BEV + ST1[i]))), Weight1[i])))
        return torch.mean(torch.cat((
            torch.matmul(probs0, tmp0),
            torch.matmul(probs1, tmp1),
            torch.matmul(probs2, tmp1),
            torch.matmul(probs3, tmp1),
            torch.matmul(probs4, tmp1)
        )))


def trainBEASTNet(nLastFile, pklName, csvName, csvNameFinal, logName, probs0, probs1, probs2, probs3, probs4, BEV, b,nIterations=10):
    """
    :param nLastFile: the number that at the end of the name of the last file
    :param pklName: name of the pickle files name without the '_number' at the end or '.pkl' ending that the function preprocess created. refred to a picke file with the whole dataset and not per sub group!!!
    :param csvName: name of the csv  file of the data you want to run the network on without the '.csv' ending file
    (if you want to run it on sub group from the original data this csv needs to be with the sub group only!!!, but the picke files doesn't!!!!)
    it needs to be in this shape (TEST_TRAIN can be 'TEST' or 'TRAIN', and Problem is the GameID):
    Ha	pHa	La	Hb	pHb	Lb	LotShapeA	LotNumA	LotShapeB	LotNumB	B_rate	TEST_TRAIN	Problem
    :param csvNameFinal: name of the final csv file the model is creating without the '.csv' ending
    :param logName:name of the log file without the '.log' ending
    :param probs0: prior value of the vector that is the bias probabilities in trial 0 are based on- numbers in float (after a softmax on the vector you get the probabilities)
    :param probs1: prior value of the vector that is the bias probabilities in trial 1 are based on- numbers in float (after a softmax on the vector you get the probabilities)
    :param probs2: prior value of the vector that is the bias probabilities in trial 2 are based on- numbers in float (after a softmax on the vector you get the probabilities)
    :param probs3: prior value of the vector that is the bias probabilities in trial 3 are based on- numbers in float (after a softmax on the vector you get the probabilities)
    :param probs3: prior value of the vector that is the bias probabilities in trial 4 are based on- numbers in float (after a softmax on the vector you get the probabilities)
    if you train a subgroup model, please change the probs0,...probs4 to be the final parameters of the training on non dominant games.
    :param BEV: prior value of BEV weight- numbers in float (for comparison ST weight is 1 always)
    :param b: prior value of the sigmoid curvature- numbers in float - higher value means curvier sigmoid

    the function trains a network the learns the parameters of the bias probabilities, the BEV weight, and the sigmoid curvature
    it prints the parameters values  every 10 games in the log file and here (so the final values will be at the end of the file)
    and at the end creates a csv file (named by the param csvNameFinal) with the prediction of the tuned BEAST in a column named 'BEASTNET'
    """
    logging.basicConfig(filename=logName + '.log', level=logging.DEBUG)
    model = BEASTNet(probs0, probs1, probs2, probs3, probs4, BEV, b)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
    Pred = torch.tensor([], dtype=torch.float64, requires_grad=True)
    true = torch.tensor([], dtype=torch.float64, requires_grad=True)
    params = ['probs0', 'probs1', 'probs2', 'probs3', 'probs4', 'BEV', 'B']
    ST1, ST0, Weight1, Weight0 = [], [], [], []
    for k1 in ['unb', 'uni', 'pes', 'sig']:
        Weight1.append('Weight_' + k1)
        ST1.append('ST_' + k1)
        Weight0.append('Weight0_' + k1)
        ST0.append('ST0_' + k1)
        for k2 in ['unb', 'uni', 'pes', 'sig']:
            Weight1.append('Weight_' + k1 + '_' + k2)
            ST1.append('ST_' + k1 + '_' + k2)
            Weight0.append('Weight0_' + k1 + '_' + k2)
            ST0.append('ST0_' + k1 + '_' + k2)
            for k3 in ['unb', 'uni', 'pes', 'sig']:
                Weight1.append('Weight_' + k1 + '_' + k2 + '_' + k3)
                ST1.append('ST_' + k1 + '_' + k2 + '_' + k3)
                Weight0.append('Weight0_' + k1 + '_' + k2 + '_' + k3)
                ST0.append('ST0_' + k1 + '_' + k2 + '_' + k3)

    for data in range(100, nLastFile, 100):
        if data == 100:
            df2 = pd.read_pickle(pklName + '_' + str(data) + '.pkl')
        else:
            df2 = df2.append(pd.read_pickle(pklName + '_' + str(data) + '.pkl'))
    if nLastFile != 100:
        df2 = df2.append(pd.read_pickle(pklName + '_' +str(nLastFile) + '.pkl'))
    else:
        df2 = pd.read_pickle(pklName + '_100.pkl')

    df = pd.read_csv(csvName + '.csv')
    df2 = df2.merge(df, on='Problem')
    nProblems2 = len(df2)
    df2.index = range(nProblems2)
    logging.debug(datetime.now())
    for t in range(nIterations):
        df2 = df2.reindex(np.random.permutation(nProblems2))
        print("########## round " + str(t) + " size of data " + str(nProblems2) + "##########")
        logging.debug("########## round " + str(t) + " size of data " + str(nProblems2) + "##########")
        for i in range(nProblems2):
            print(str(i))
            logging.debug(str(i))
            if df2['TEST_TRAIN'][i] == 'TRAIN':
                optimizer.zero_grad()
                Pred = torch.cat(
                    (Pred, torch.reshape(model(
                        df2['BEVb'][i] - df2['BEVa'][i],
                        [torch.reshape(torch.tensor(df2[j][i], dtype=torch.float64), (1, len(df2[j][i]))) for j in ST0],
                        [torch.reshape(torch.tensor(df2[j][i], dtype=torch.float64), (len(df2[j][i]), 1)) for j in
                         Weight0],
                        [torch.reshape(torch.tensor(df2[j][i], dtype=torch.float64), (1, len(df2[j][i]))) for j in ST1],
                        [torch.reshape(torch.tensor(df2[j][i], dtype=torch.float64), (len(df2[j][i]), 1)) for j in
                         Weight1]), (1,))))
                true = torch.cat(
                    (true, torch.reshape(torch.tensor(df2['B_rate'][i], dtype=torch.float64, requires_grad=True), (1,))))
                if (i + 1) % 10 == 0:
                    loss = criterion(Pred, true)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    j = 0
                    for param in model.parameters():
                        print(params[j] + ' ' + str(param))
                        j += 1
                    Pred = torch.tensor([], dtype=torch.float64, requires_grad=True)
                    true = torch.tensor([], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        j = 0
        for param in model.parameters():
            print(params[j] + ' ' + str(param))
            logging.debug(params[j] + ' ' + str(param))
            j+=1
        df2['BEASTNET'] = df2.apply(lambda x: (model(x['BEVb'] - x['BEVa'],
                                                     [torch.reshape(torch.tensor(x[j], dtype=torch.float64),
                                                                    (1, len(x[j]))) for j in ST0],
                                                     [torch.reshape(torch.tensor(x[j], dtype=torch.float64),
                                                                    (len(x[j]), 1)) for j in Weight0],
                                                     [torch.reshape(torch.tensor(x[j], dtype=torch.float64),
                                                                    (1, len(x[j]))) for j in ST1],
                                                     [torch.reshape(torch.tensor(x[j], dtype=torch.float64),
                                                                    (len(x[j]), 1)) for j in
                                                      Weight1]).detach().numpy()), axis=1)
        pd.read_csv(csvName + '.csv').merge(df2[['BEASTNET', 'Problem']], on='Problem').to_csv(
            csvNameFinal + '.csv', index=False)


if __name__ == '__main__':
    #preprocess(csvName='RealData270', pklName='RealData270')
    """
    run preprocessing only *once*, *on the whole dataset*. then you can train many models as you wish, per subgroup or for all of the data.
    you will receive  pickles files of the processed dataset.
    """
    trainBEASTNet(nLastFile=270, pklName='RealData270', csvName='RealData270_dom', csvNameFinal='RealData270_cpc_13k_dom', logName='RealData270_cpc_13k_dom',
               probs0=[1.3971,1.1172,1.2091,1.2767], probs1=[2.0527,1.1718,1.2514,1.2241], probs2=[2.1565,1.1759,1.2526,1.2150],
               probs3=[2.2635,1.1786,1.2523,1.2057], probs4=[2.3740,1.1798,1.2502,1.1960], BEV=0, b=0.81695)
    """
    those probs0,...probs1 values are for the probabilities of the original BEAST model.
    if you want to run a subgroup model. please train the model first on non dominant problems, and then take the parameters it learned from the log file, and  put them in probs0,...probs4 and run the model on the subgroup.
    when you run a model on a subgroup/non_dom/dom choice tasks, the csv must include only those choice tasks. while the pickle file will include all of the dataset
    for getting a clustered model please run deep_clustering_kmeans.py which produced a clustered dataset (with column 'cluster') and then run a different BEASTNet model per cluster, you will need to create a seperate csv file per cluster as any other subgroup
    """

