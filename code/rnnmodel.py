import math
import torch
import torch.nn as nn

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init


class BIRNN(nn.Module):
    """
    Bidirectional RNN that is initialized with the model myRNN.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, pretrained_ebd, dropout):
        """
        :param vocab_size: size of vocabulary (len(text.vocab))
        :param embed_size: embed layer size
        :param hidden_size: hidden layer size
        :param pretrained_ebd: gloVe pretrained model
        :param dropout: dropout rate (0-1)
        """
        super(BIRNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.emb = self.emb.from_pretrained(pretrained_ebd, freeze=False)

        # initializes RNN with myRNN
        self.RNN = myRNN(50,
                         hidden_size=hidden_size,
                         batch_first=True,
                         bidirectional=True,
                         dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, 2)

    def forward(self, text1):
        emb = self.emb(text1)

        o_n, h_n, = self.RNN(emb)   # RNN embeds
        hidden_out = torch.cat((h_n[0, :, :], h_n[1, :, :]), 1) # hidden layer vectors
        out = self.linear(hidden_out)
        return out

class myRNN(Module):
    """implementation of RNN as a pytorch module
    """
    def __init__(self,  input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(myRNN, self).__init__()

        # model settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2

        self._all_weights = []

        # setup parameters
        for direction in range(num_directions):
            layer_input_size = input_size 
            w_ih = Parameter(torch.Tensor(hidden_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
            b_ih = Parameter(torch.Tensor(hidden_size))
            b_hh = Parameter(torch.Tensor(hidden_size))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            suffix = '_reverse' if direction == 1 else ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            if bias:
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(0, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

        # using continuous memory
        self.flatten_parameters()

        # set weight by uniform distribution
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)        

    def flatten_parameters(self):
        """allocate parameters into a continuous chunk of memory
        reference: pytorch's RNN flatten_parameters function
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return
        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn
            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode('RNN_TANH'), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def forward(self, input, hx=None):
        """
        forward pass:
        - initialize hidden layers to 0 before calculation,
        - initialize the target vectors
        - use rnn_tanh to calculate output
        """
        max_batch_size = input.size(0)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
        cal = torch._C._VariableFunctions.rnn_tanh
        result = cal(input, hx, 
                        self._flat_weights, self.bias, self.num_layers,
                        self.dropout, self.training, self.bidirectional, 
                        self.batch_first)


        output = result[0]              # final output
        last_hidden_state = result[1]   # final hidden layer vector

        return output, last_hidden_state

    @property
    def _flat_weights(self):
        return [
            p for layerparams in self.all_weights
            for p in layerparams
        ]

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]

