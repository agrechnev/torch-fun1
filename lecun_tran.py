# Created by  IT-JIM  2021
# Here I fool around with LeCun's transformer
# And try to solve the issues with torchtext version incompatibilities

import sys

import numpy as np
import torch
import torch.utils.data
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
            d_xq = d_xk = d_xv = d_model  # All same
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
        # batch_size = q.size(0)
        # k_length = k.size(-2)
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        q = q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        a = torch.nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        # Get the weighted average of the values
        h = torch.matmul(a, v)
        return h, a

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, x_q, x_k, x_v):
        batch_size, seq_length, dim = x_q.size()

        # After transforming, split into num_heads
        q = self.split_heads(self.w_q(x_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        k = self.split_heads(self.w_k(x_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        v = self.split_heads(self.w_v(x_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)

        # Calculate the attention weights for each of the heads
        h_cat, a = self.scaled_dot_product_attention(q, k, v)
        # Put all the heads back together by concat
        h_cat = self.group_heads(h_cat, batch_size)
        h = self.w_h(h_cat)
        return h, a


########################################################################################################################
def main_stupid1():
    """Test kvq selection"""
    print('haha')
    mha = MultiHeadAttention(512, 8)
    k = torch.tensor(
        [[10, 0, 0],
         [0, 10, 0],
         [0, 0, 10],
         [0, 0, 10]],
        dtype=torch.float32).view(1, 1, 4, 3)

    v = torch.tensor(
        [[1, 0, 0],
         [10, 0, 0],
         [100, 5, 0],
         [1000, 6, 0]],
        dtype=torch.float32).view(1, 1, 4, 3)

    q = torch.tensor(
        [[0, 0, 10]],
        dtype=torch.float32).view(1, 1, 1, 3)

    out, attn = mha.scaled_dot_product_attention(q, k, v)
    print('out=', out)
    print('attn=', attn)


########################################################################################################################
def main_stupid2():
    """This time, the dataset. Cannot understand the new torchtext!"""
    dset_train, dset_val = torchtext.datasets.IMDB(root='/home/seymour/data')
    print('dset_train', len(dset_train))
    print('dset_val', len(dset_val))
    batch_size = 164
    loader_train = torch.utils.data.DataLoader(dset_train, batch_size=batch_size)
    loader_val = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    print('loader_train', len(loader_train))
    print('loader_val', len(loader_val))
    # x, y = next(iter(loader_val))
    for x, y in loader_train:
        print('x=', x)


########################################################################################################################
def main_stupid3():
    import torchtext.legacy.data as data
    import torchtext.legacy.datasets as datasets
    max_len = 200
    text = data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
    label = data.LabelField(sequential=False, dtype=torch.long)
    # datasets.IMDB.download('/home/seymour/data')
    ds_train, ds_test = datasets.IMDB.splits(text, label, path='/home/seymour/data/IMDB/aclImdb')
    print('ds_train', len(ds_train))
    print('ds_test', len(ds_test))



########################################################################################################################
if __name__ == '__main__':
    main_stupid3()
