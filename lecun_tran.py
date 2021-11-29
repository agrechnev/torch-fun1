# Created by  IT-JIM  2021
# Here I fool around with LeCun's transformer
# And try to solve the issues with torchtext version incompatibilities

import sys

import numpy as np
import torch
import torchtext


########################################################################################################################
def print_it(a, name: str = ''):
    print(name, a.shape, a.dtype, a.min(), a.mean(), a.max())


########################################################################################################################
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, p=0, d_input=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model   # All same
        else:
            d_kq, d_xk, d_xv = d_input  # Not used in this code !!!

        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. They will be split into number of heads
        self.w_q = torch.nn.Linear(d_xq, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_xk, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.w_h = torch.nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v):
        batch_size = q.size(0)
        k_length = k.size(-2)
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        q = q / np.sqrt(self.d_k)                      # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))    # (bs, n_heads, q_length, k_length)

        a = torch.nn.Softmax(dim=-1)(scores)              # (bs, n_heads, q_length, k_length)
        # Get the weighted average of the values
        h = torch.matmul(a, v)
        return h, a


########################################################################################################################
def main():
    # dset_train, dset_val = torchtext.datasets.IMDB(root='/home/seymour/data')
    # print(type(dset_train), len(dset_train))
    # print(type(dset_val), len(dset_val))

    print('haha')
    mha = MultiHeadAttention(512, 8)



########################################################################################################################
if __name__ == '__main__':
    main()