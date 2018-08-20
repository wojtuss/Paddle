#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
from .. import core
from ..framework import Program
from ..executor import global_scope

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
from .. import core
from ..framework import Program
from ..executor import global_scope


class InferenceTranspiler(object):
    '''
    Convert the fluid program to optimized inference program.

    There are several optimizations:

      - fuse convolution and batch normalization
      - fuse batch normalization and relu (MKLDNN only)

    Examples:

    .. code-block:: python

        # As InferenceTranspiler will modify the original program,
        # please clone before use it.
        inference_transpiler_program = program.clone()
        t = fluid.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)
    '''

    def transpile(self, program, place, scope=None):
        '''
        Run the transpiler.

        Args:
            program (Program): program to transpile
            place (Place): inference place
            scope (Scope|None): inference Scope
        '''
        print("--- Transpiler ---")
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")
        if not isinstance(place, core.CPUPlace) and not isinstance(
                place, core.CUDAPlace):
            raise TypeError("place should be as CPUPlace/CUDAPlace type")
        if scope is None:
            scope = global_scope()
        if not isinstance(scope, core.Scope):
            raise TypeError("scope should be as Scope type or None")
        self._fuse_batch_norm(program, place, scope)
        self._fuse_relu_mkldnn(program)

        i = 0
        print("Before")
        while i < len(self.block.ops):
            current_op = self.block.ops[i]
            print("___{}".format(current_op.type))
            i = i + 1
        self.fuse_fc_gru_mkldnn(program)
        i = 0
        print("After")
        while i < len(self.block.ops):
            current_op = self.block.ops[i]
            print("___{}".format(current_op.type))
            i = i + 1

    def _fuse_relu_mkldnn(self, program):
        '''
        Transpile the program by fused relu activation for MKLDNN program.

        Relu activation following batch norm OP can be fused by adding
        :math:`fuse_with_relu` attribute to batch norm OP.

        The result of fuse is:

        - before:

          - batch_norm->relu->any_other_op

        - after:

          - batch_norm->any_other_op

        :param program: program to transpile
        :type program: Program
        '''
        use_mkldnn = bool(os.getenv("FLAGS_use_mkldnn", False))
        if not use_mkldnn:
            return

        self.block = program.block(0)

        i = 0
        while i < len(self.block.ops) - 1:
            current_op = self.block.ops[i]
            if current_op.type in ['batch_norm']:
                next_op = self.block.ops[i + 1]
                if next_op.type == 'relu':
                    # modify bnorm OP to include relu
                    current_op.set_attr("fuse_with_relu", True)
                    # remove relu OP
                    self.block._remove_op(i + 1)
            i = i + 1

        self._remove_unused_var()
        # TODO(luotao): use clone() method to flush the program.desc in force,
        # since some large program.desc will not be flushed immediately.
        # And a better solution will be considered later.
        program = program.clone()

    def _fuse_batch_norm(self, program, place, scope):
        '''
        Transpile the program by fused batch normalization.

        The batch normalization followed the convolution or fully connected layer
        can be integrated with them. Doing so will give us a forward acceleration,
        especially in environments like mobile or embedded.

        For input :math:`X`:

        - Conv process:        :math:`X = input * W + bias`
        - Batch norm process:  :math:`X' = (X - mean) / std`
        - Scale Process:       :math:`Y = a * X' + b`

        After fuse into one operation:

        .. math::

            Y &= (input * W + bias - mean) / std * a + b \\\\
              &= input * a * W / std + ((bias - mean) / std * a + b)

        The operator transformation is:

        - before:

          - conv->batch_norm->any_other_op (bias == 0)
          - conv->elementwise_add->batch_norm->any_other_op (bias != 0)

        - after:

          - conv->elementwise_add->any_other_op

        The transpile stages are:

        1. insert elementwise_add op when bias == 0.
        2. fuse the batch_norm's parameters to conv and elementwise_add operators.
        3. remove batch_norm ops which are not used in any other ops.
        4. adjust the input of any_other_op to be the output of elementwise_add operator.
        5. remove unused variables.

        Args:
            program (Program): program to transpile
            place (Place): inference place
            scope (Scope): inference Scope

        '''
        self.scope = scope
        self.place = place
        self.block = program.block(0)
        self.input_map = {}  # store the input names should be adjusted

        i = 0
        while i < len(self.block.ops) - 2:
            current_op = self.block.ops[i]
            # TODO(luotao1): consider only conv2d now. fc would be deltt later.
            if current_op.type in ['conv2d']:
                # TODO(luotao1): consider single chain network now.
                # For branch network, we counldn't use block.ops[i + 1] as
                # the judgment condition.
                next_op = self.block.ops[i + 1]
                # conv2d without bias
                if (next_op.type == 'batch_norm'):
                    # insert bias op
                    bias_op = self._insert_bias_op(i + 1, current_op, next_op)
                    # fuse batch_norm
                    self._fuse_param(current_op, next_op, bias_op, 0)
                    # remove batch_norm_op
                    self.block._remove_op(i + 2)
                    i = i + 1
                # conv2d with bias, the next_op.type is elementwise_add
                elif (next_op.type == 'elementwise_add'):
                    next_next_op = self.block.ops[i + 2]
                    if (next_next_op.type == 'batch_norm'):
                        # fuse batch_norm
                        self._fuse_param(current_op, next_next_op, next_op, 1)
                        # remove batch_norm_op
                        self.block._remove_op(i + 2)
                        i = i + 1
            i = i + 1

        self._adjust_input()
        self._remove_unused_var()
        # TODO(luotao): use clone() method to flush the program.desc in force,
        # since some large program.desc will not be flushed immediately.
        # And a better solution will be considered later.
        program = program.clone()

    def fuse_fc_gru_mkldnn(self, program):
        '''
        Transpile the program by fusing FC+GRU layers with the MKL-DNN GRU.

        The GRU following a FC layer can be replaced by the MKL-DNN GRU.
        The FC's MUL op weight input 'Y' has to be transformed into the
        MKL-DNN-based GRU input 'WeightX'.

        The operator transformation is:

        - before:

          - FC (MUL->elementwise_add) -> GRU -> any_other_op

        - after:

          - GRU -> any_other_op

        The transpile stages are:

        1. insert a new MKL-DNN-based GRU operator with `WeightX` input
           (weights) taken from the MUL's input 'Y' (weights),
        2. fuse the parameters of MUL and GRU,
        3. remove the MUL, elementwise_add and the old GRU operators,
        4. make the input of the deleted MUL operator to be the input of the
           new GRU operator,
        5. remove unused variables,

        Args:
            program (Program): program to transpile

        '''
        use_mkldnn = bool(os.getenv("FLAGS_use_mkldnn", False))
        if not use_mkldnn:
            return

        self.block = program.block(0)

        i = 0
        while i < len(self.block.ops):
            # find a gru op
            if self.block.ops[i].type == 'gru':
                gru_op = self.block.ops[i]
                gru_idx = i
                add_idx = -1
                mul_idx = -1
                # find the preceding elementwise_add op
                for j in reversed(range(gru_idx)):
                    if self.block.ops[j].type == 'elementwise_add':
                        gru_input_names = gru_op.input_arg_names
                        add_out_name = self.block.ops[j].output_arg_names[0]
                        if self.block.ops[j].output_arg_names[
                                0] in gru_op.input_arg_names:
                            add_op = self.block.ops[j]
                            add_idx = j
                            break
                if add_idx < 0:
                    i += 1
                    continue

                # find the preceding mul op
                for j in reversed(range(add_idx)):
                    if self.block.ops[j].type == 'mul':
                        mul_out_name = self.block.ops[j].output_arg_names[0]
                        if self.block.ops[j].output_arg_names[
                                0] in add_op.input_arg_names:
                            mul_op = self.block.ops[j]
                            mul_idx = j
                            break
                if mul_idx < 0:
                    i += 1
                    continue

                # create and insert a new gru op
                gru_op_new = self._insert_gru_op(gru_idx + 1, mul_op, gru_op)
                # fuse mul's and gru's parameters
                self._fuse_gru(mul_op, gru_op_new)
                # Add two bias
                self._fuse_gru_bias(add_op, gru_op_new)
                # remove the old operators
                self.block._remove_op(gru_idx)
                self.block._remove_op(add_idx)
                self.block._remove_op(mul_idx)
                # restart scanning for gru from the deleted mul's index
                i = mul_idx
            i += 1

        self._adjust_input()
        self._remove_unused_var()
        program = program.clone()

    # ====================== private transpiler functions =====================
    def _insert_bias_op(self, index, current_op, bn_op):
        '''
        Construct elementwise_add operator for adding bias
        and insert it into program.

        :param index: insert location of bias_op
        :type index: Int
        :param current_op: current operator (conv or fc)
        :type current_op: Operator
        :param bn_op: batch norm operator
        :type bn_op: Operator
        :return: bias_op
        :rtype: Operator
        '''
        # The input of bias_op is current_op's output and Bias of bn_op
        # The output of bias_op is bn_op's output
        x_var = self.block.var(current_op.output("Output")[0])
        y_var = self.block.var(bn_op.input("Bias")[0])
        out_var = self.block.var(bn_op.output("Y")[0])

        bias_op = self.block._insert_op(
            index,
            type="elementwise_add",
            inputs={"X": x_var,
                    "Y": y_var},
            outputs={"Out": out_var},
            attrs={"axis": 1})  # dim_start=1
        return bias_op

    def _insert_gru_op(self, index, mul_op, gru_op):
        '''
        Construct a new GRU operator by copying the old GRU and adding the
        'WeightX' input taken from the MUL's input 'Y'.

        :param index: insert location of GRU
        :type  index: Int
        :param mul_op: MUL operator to copy weights from
        :type  mul_op: Operator
        :param gru_op: GRU operator to be copied
        :type  gru_op: Operator
        :return: gru_op_new
        :type:   Operator
        '''

        def get_op_inputs(op, names):
            result = {}
            for name in names:
                if op.input(name):
                    result[name] = self.block.var(op.input(name)[0])
            return result

        def get_op_outputs(op, names):
            result = {}
            for name in names:
                result[name] = self.block.var(op.output(name)[0])
            return result

        gru_inputs = get_op_inputs(gru_op, ['Bias', 'Weight', 'H0'])
        gru_inputs['WeightX'] = self.block.var(mul_op.input('Y')[0])
        gru_inputs['WeightH'] = gru_inputs.pop('Weight')
        gru_inputs['Input'] = self.block.var(mul_op.input('X')[0])
        gru_outputs = {}
        gru_outputs['Hidden'] = self.block.var(gru_op.output('Hidden')[0])
        gru_attrs = gru_op.all_attrs()
        gru_attrs['use_mkldnn'] = True
        gru_attrs['stack_level'] = 1
        gru_attrs['is_bidirection'] = False

        gru_op_new = self.block._insert_op(
            index,
            type='gru_fused',
            inputs=gru_inputs,
            outputs=gru_outputs,
            attrs=gru_attrs)
        return gru_op_new

    def _update_param(self, op, old_param_name, new_param, suffix):
        # For the sake of remaining the original variables the same as before,
        # create new variables in scope to store the new parameters.
        old_param_name = old_param_name[0]
        old_var = self.block.vars[old_param_name]
        new_param_name = old_param_name + '_fuse_' + suffix
        new_var = self.block.create_parameter(
            name=new_param_name.encode('ascii'),
            type=old_var.type,
            dtype=old_var.dtype,
            shape=old_var.shape)
        op.rename_input(old_param_name, new_param_name)
        self.scope.var(new_param_name)

        tensor = self.scope.find_var(new_param_name).get_tensor()
        tensor.set(np.array(new_param), self.place)

    def _fuse_param(self, current_op, bn_op, bias_op, with_bias):
        '''
        fuse the batch_norm_op' parameters to current_op (conv or fc)

        :param current_op: current operator (conv or fc)
        :type current_op: Operator
        :param bn_op: batch norm operator
        :type bn_op: Operator
        :param bias_op: elementwise_add operator for adding bias
        :type bias_op: Operator
        :param with_bias: If current operator has bias, with_bias = 1; otherwise 0.
        :type with_bias: Int
        '''

        def _load_param(param_name):
            return np.array(self.scope.find_var(param_name[0]).get_tensor())

        bias_bn = _load_param(bn_op.input("Bias"))  #Bias
        scale_bn = _load_param(bn_op.input("Scale"))  #Scale
        mean_bn = _load_param(bn_op.input("Mean"))  #Mean
        var_bn = _load_param(bn_op.input("Variance"))  #Variance

        # TODO(luotao1): consider only conv2d now. fc would be delt later.
        current_param = _load_param(current_op.input("Filter"))
        std_bn = np.float32(np.sqrt(np.add(var_bn, 1e-5)))
        tmp = np.float32(np.divide(scale_bn, std_bn))

        # add bias of batch_norm_op to conv2d
        if with_bias:
            bias = _load_param(bias_op.input("Y"))
        else:
            bias = np.zeros(bias_bn.shape)
        bias = np.float32(
            np.add(np.multiply(np.subtract(bias, mean_bn), tmp), bias_bn))

        # re-compute weight of conv2d
        tmp = tmp.reshape(tmp.shape[0], -1)
        dst_param = current_param.reshape((tmp.shape[0], -1))
        dst_param = np.float32(np.multiply(dst_param, tmp))
        dst_param = dst_param.reshape(current_param.shape)

        # update parameters
        self._update_param(current_op,
                           current_op.input("Filter"), dst_param, 'bn')
        self._update_param(bias_op, bias_op.input("Y"), bias, 'bn')

        # collect the renamed input
        self.input_map[bn_op.output("Y")[0]] = bias_op.output("Out")[0]

    def _fuse_gru(self, mul_op, gru_op):
        '''
        fuse the MUL's and GRU's weight parameters

        :param mul_op: MUL op to take weights from
        :type  mul_op: Operator
        :param gru_op: GRU op to put the weights to
        :type  gru_op: Operator
        '''
        # get data from the mul op weights
        weight_x = np.array(
            self.scope.find_var(mul_op.input('Y')[0]).get_tensor())
        # update weight parameters
        self._update_param(gru_op, gru_op.input('WeightX'), weight_x, 'gru')
        # save weight names for update
        self.input_map[gru_op.input('Input')[0]] = mul_op.input('X')[0]

    def _fuse_gru_bias(self, add_op, gru_op):
        '''
        fuse the FC's and GRU's bias parameters

        :param add_op: Elementwise_add op to take weights from
        :type  mul_op: Operator
        :param gru_op: GRU op to put the weights to
        :type  gru_op: Operator
        '''
        # get data from the add op and gru op weights
        bias_fc = np.array(
            self.scope.find_var(add_op.input('Y')[0]).get_tensor())
        bias_gru = np.array(
            self.scope.find_var(gru_op.input('Bias')[0]).get_tensor())
        # Add two bias to get a new bias for fused gru
        bias_new = np.add(bias_fc, bias_gru)
        # update weight parameters
        self._update_param(gru_op, gru_op.input('Bias'), bias_new, 'gru')

    def _adjust_input(self):
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            for input_arg in current_op.input_arg_names:
                if input_arg in self.input_map:
                    current_op.rename_input(input_arg,
                                            self.input_map[input_arg])

    def _remove_unused_var(self):
        '''
        remove unused varibles in program
        '''
        args = []
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            args += current_op.input_arg_names
            args += current_op.output_arg_names
        args = list(set(args))  # unique the input and output arguments

        for var in list(self.block.vars.keys()):
            if var not in args:
                self.block._remove_var(var)
