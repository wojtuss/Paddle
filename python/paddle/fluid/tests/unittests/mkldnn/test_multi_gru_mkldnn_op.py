#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_fusion_gru_op import fusion_gru, ACTIVATION


def multi_gru(
        x,  # T x M
        lod,  # 1 x N
        h0,  # N x D
        wx,  # M x 3D
        wh,  # D x 3D
        bias,  # 1 x 3D
        origin_mode,
        layers):
    act_state = ACTIVATION['tanh']
    act_gate = ACTIVATION['sigmoid']
    input = x
    for i in range(0, layers * 2, 2):
        _, _, _, gru1_out = fusion_gru(input, lod, h0[i], wx[i], wh[i], bias[i],
                                       False, origin_mode, act_state, act_gate)
        _, _, _, gru2_out = fusion_gru(input, lod, h0[i + 1], wx[i + 1],
                                       wh[i + 1], bias[i + 1], True,
                                       origin_mode, act_state, act_gate)
        input = np.concatenate((gru1_out, gru2_out), axis=1)
    return input


class TestMultiGRUOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "multi_gru"
        self.dtype = "float32"
        self.lod = [[2, 4, 3]]
        self.M = 3
        self.D = 5
        self.with_h0 = False
        self.with_bias = False
        self.layers = 1
        self.origin_mode = False
        self._cpu_only = True
        self.set_confs()

        T = sum(self.lod[0])
        N = len(self.lod[0])
        x = np.random.rand(T, self.M).astype('float32')
        h0 = np.random.rand(
            N, self.D).astype('float32') if self.with_h0 else np.zeros(
                (N, self.D), dtype='float32')
        #  self.inputs = {
        #  'X': (x, self.lod),
        #  'WeightX': [],
        #  'WeightH': [],
        #  'Bias': [],
        #  'H0': []
        #  }

        #  wx = np.ndarray()
        #  wh = np.ndarray()
        #  bias = np.ndarray()
        #  h0 = np.ndarray()
        wx = []
        wh = []
        bias = []
        h0 = []

        for i in range(self.layers * 2):
            wx.append(np.random.rand(self.M, 3 * self.D).astype('float32'))
            wh.append(np.random.rand(self.D, 3 * self.D).astype('float32'))
            bias.append(
                np.random.rand(1, 3 * self.D).astype('float32')
                if self.with_bias else np.zeros(
                    (1, 3 * self.D), dtype='float32'))
            h0.append(np.zeros((N, self.D), dtype='float32'))

        hidden = multi_gru(x, self.lod, h0, wx, wh, bias, self.origin_mode,
                           self.layers)

        self.inputs = {
            'X': (x, self.lod),
            'WeightX': [('wx' + str(i), wx[i]) for i in range(self.layers * 2)],
            'WeightH': [('wh' + str(i), wh[i]) for i in range(self.layers * 2)]
            #  'WeightX': tuple(wx[i] for i in range(self.layers * 2)),
            #  'WeightH': tuple(wh[i] for i in range(self.layers * 2))
            #  'WeightX': [wx[i] for i in range(self.layers * 2)],
            #  'WeightH': [wh[i] for i in range(self.layers * 2)]
        }

        if self.with_bias:
            #  self.inputs['Bias'] = tuple(bias[i] for i in range(self.layers * 2))
            self.inputs['Bias'] = [('b' + str(i), bias[i])
                                   for i in range(self.layers * 2)]
        if self.with_h0:
            self.inputs['H0'] = h0[0]

        self.outputs = {'Hidden': (hidden, self.lod)}

        self.attrs = {
            'activation': 'tanh',
            'gate_activation': 'sigmoid',
            'layers': self.layers,
            'origin_mode': self.origin_mode,
            'use_mkldnn': True
        }

    def test_check_output(self):
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(check_dygraph=False)


#  class TestMultiGRUMKLDNNOpNoH0(TestMultiGRUOp):
#  def set_confs(self):
#  self.with_h0 = False

#  class TestMultiGRUMKLDNNOpNoBias(TestMultiGRUOp):
#  def set_confs(self):
#  self.with_bias = False

#  class TestMultiGRUMKLDNNOpLayers2(TestMultiGRUOp):
#  def set_confs(self):
#  self.layers = 2

#  class TestMultiGRUMKLDNNOpLayers3(TestMultiGRUOp):
#  def set_confs(self):
#  self.layers = 3

#  class TestMultiGRUMKLDNNOpOriginMode(TestMultiGRUOp):
#  def set_confs(self):
#  self.origin_mode = True

if __name__ == "__main__":
    unittest.main()
