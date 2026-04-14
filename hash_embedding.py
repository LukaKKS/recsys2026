from __future__ import unicode_literals, division

import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import normal


class HashFamily():
    r"""Universal hash family as proposed by Carter and Wegman.

    .. math::

            \begin{array}{ll}
            h_{{a,b}}(x)=((ax+b)~{\bmod  ~}p)~{\bmod  ~}m \ \mid p > m\\
            \end{array}

    Args:
        bins (int): Number of bins to hash to. Better if a prime number.
        mask_zero (bool, optional): Whether the 0 input is a special "padding" value to mask out.
        moduler (int,optional): Temporary hashing. Has to be a prime number.
    """

    def __init__(self, bins, mask_zero=False, moduler=None):
        if moduler and moduler <= bins:
            raise ValueError("p (moduler) should be >> m (buckets)")

        self.bins = bins
        self.moduler = moduler if moduler else self._next_prime(np.random.randint(self.bins + 1, 2**32))
        self.mask_zero = mask_zero

        # do not allow same a and b, as it could mean shifted hashes
        self.sampled_a = set()
        self.sampled_b = set()

    def _is_prime(self, x):
        """Naive is prime test."""
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def _next_prime(self, n):
        """Naively gets the next prime larger than n."""
        while not self._is_prime(n):
            n += 1

        return n

    def draw_hash(self, a=None, b=None):
        """Draws a single hash function from the family."""
        if a is None:
            while a is None or a in self.sampled_a:
                a = np.random.randint(1, self.moduler - 1)
                assert len(self.sampled_a) < self.moduler - 2, "please give a bigger moduler"

            self.sampled_a.add(a)
        if b is None:
            while b is None or b in self.sampled_b:
                b = np.random.randint(0, self.moduler - 1)
                assert len(self.sampled_b) < self.moduler - 1, "please give a bigger moduler"

            self.sampled_b.add(b)

        if self.mask_zero:
            return lambda x: ((a * x + b) % self.moduler) % (self.bins - 1) + 1
        else:
            return lambda x: ((a * x + b) % self.moduler) % self.bins

    def draw_hashes(self, n, **kwargs):
        """Draws n hash function from the family."""
        return [self.draw_hash() for i in range(n)]

    def draw_hashes_tensor(self, n, device='cpu'):
        multipliers = []
        adders = []

        for _ in range(n):
            a = None
            b = None
            if a is None:
                while a is None or a in self.sampled_a:
                    a = np.random.randint(1, self.moduler - 1)
                    assert len(self.sampled_a) < self.moduler - 2, "please give a bigger moduler"

                self.sampled_a.add(a)
            if b is None:
                while b is None or b in self.sampled_b:
                    b = np.random.randint(0, self.moduler - 1)
                    assert len(self.sampled_b) < self.moduler - 1, "please give a bigger moduler"

                self.sampled_b.add(b)
            
            multipliers.append(a)
            adders.append(b)
        
        multipliers_tensor = torch.tensor(multipliers, device=device)
        multipliers_tensor = multipliers_tensor.view(-1, 1, 1)
        adders_tensor = torch.tensor(adders, device=device)
        adders_tensor = adders_tensor.view(-1, 1, 1)
        return multipliers_tensor, adders_tensor


class HashEmbedding(nn.Module):
    r"""Type of embedding which uses multiple hashes to approximate an Embedding layer using less parameters.

    This module is a new Embedding module that compresses the number of parameters. They are a
    generalization of vanilla Embeddings and the `hashing trick`. For more details, check Svenstrup,
    Dan Tito, Jonas Hansen, and Ole Winther. "Hash embeddings for efficient word
    representations." Advances in Neural Information Processing Systems. 2017.

    For each elements (usually word indices) in the input (mini_batch, sequence_length) the default
    computations are:

    .. math::

            \begin{array}{ll}
            H_i = E_{D_2^i(D_1(w)))} \ \forall i=1...k\\
            c_w = (H_1(w), ..., H_k(w))^T\\
            p_w = P_{D_1(w)}\\
            \hat{e}_w = p_w \cdot c_w\\
            e_w = \mathrm{concatenate}(\hat{e}_w,p_w)\\
            \end{array}

    where :math:`w:[0,T]` is the element of the input (word index), :math:`D_1:[0,T)\to [0,K)`
    is the token to ID hash/dictionnary, :math:`D_2:[0,K)\to[0,B)` is the ID to Bucket hash,
    :math:`E:\mathbb R^{B*d}` is the shared pool of embeddings, :math:`c_w:\mathbb R^{k*d}` contains all
    the vector embeddings to which :math:`w` maps, :math:`e_w:\mathbb R^{d+k}` is the outputed word
    embedding for :math:`w`.

    Args:
        num_embeddings (int): the number of different embeddings. K in the paper.
            Higher increases possible vocabulary size.
        embedding_dim (int): the size of each embedding vector in the shared pool. d in the paper.
            Higher improves downstream task for fixed vocabulary size.
        num_buckets (int,optional): the size of the shared pool of embeddings. B in the paper.
            Higher improves approximation quality. Typically num_buckets * 10 < num_embeddings.
        num_hashes (int,optional): the number of different hash functions. k in the paper.
            Higher improves approximation quality. Typically in [1,3].
        train_sharedEmbed (bool,optional): whether to train the shared pool of embeddings E.
        train_weight (bool,optional): whether to train the importance parameters / weight P.
        append_weight (bool,optional): whether to append the importance parameters / weight pw.
        aggregation_mode ({"sum","median","concatenate"},optional): how to aggregate the (weighted) component
            vectors of the different hashes. Sum should be the same as mean (because learnable parameters,
            can learn to divide by n)
        mask_zero (bool, optional): whether the 0 input is a special "padding" value to mask out.
        seed (int, optional): sets the seed for generating random numbers.
        oldAlgorithm (bool, optional): whether to use the algorithm in the paper rather than the improved version.
            I do not recommend to set to true besides for comparaison.

    Attributes:
        shared_embeddings (nn.Embedding): the shared pool of embeddings of shape (num_buckets, embedding_dim).
            E in the paper.
        importance_weights (nn.Embedding): the importance parameters / weight of shape
            (num_embeddings, num_hashes). P in the paper.
        output_dim (int): effective outputed number of embeddings.

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, output_dim)`, output_dim is the effective embedding dim.

    Examples::
        >>> # an HashEmbedding module containing approximating nn.Embedding(10, 5) with less param
        >>> embedding = HashEmbedding(10,5,append_weight=False)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.3988  0.5234 -0.6148  0.3000 -1.5525
           0.1259  0.4142 -0.8613  0.3018 -1.3547
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.4672 -0.7971 -0.2009  0.7829 -0.9448

        (1 ,.,.) =
        1.00000e-04 *
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.0878 -0.1680  0.3896  0.5288 -0.2060
           0.1259  0.4142 -0.8613  0.3018 -1.3547
          -0.3098  0.0357 -0.7532 -0.1216 -0.0366
        [torch.FloatTensor of size 2x4x5]

        >>> # example with mask_zero which corresponds to padding_idx=0
        >>> embedding = HashEmbedding(10,5,append_weight=False, mask_zero=True)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.0000  0.0000  0.0000  0.0000  0.0000
          -1.4941 -1.3775 -0.5797  1.3187 -0.0555
           0.0000  0.0000  0.0000  0.0000  0.0000
          -0.7717 -0.5569 -0.1397  1.1101 -0.0939
        [torch.FloatTensor of size 1x4x5]
    """

    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, num_hashes=2, train_sharedEmbed=True,
                 train_weight=True, append_weight=True, aggregation_mode='sum', mask_zero=False, seed=None, oldAlgorithm=False, offsets=0, device='cpu'):
        super(HashEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        defaultNBuckets = (num_embeddings * self.num_hashes) // (self.embedding_dim)
        self.num_buckets = num_buckets if num_buckets is not None else defaultNBuckets
        self.train_sharedEmbed = train_sharedEmbed
        self.train_weight = train_weight
        self.append_weight = append_weight
        self.padding_idx = 0 if mask_zero else None
        self.seed = seed
        self.oldAlgorithm = oldAlgorithm

        hashFamily = HashFamily(self.num_buckets, mask_zero=mask_zero)
        self.hashes = hashFamily.draw_hashes(self.num_hashes)
        self.multiplers_tensor, self.adders_tensor = hashFamily.draw_hashes_tensor(self.num_hashes, device)
        self.bins = self.num_buckets
        self.moduler = hashFamily.moduler
        self.short_head_offsets = offsets

        if aggregation_mode == 'sum':
            self.aggregate = lambda x: torch.sum(x, dim=-1)
        elif aggregation_mode == 'concatenate':
            # little bit quicker than permute/contiguous/view
            self.aggregate = lambda x: torch.cat([x[:, :, :, i] for i in range(self.num_hashes)], dim=-1)
        elif aggregation_mode == 'median':
            print('median')
            self.aggregate = lambda x: torch.median(x, dim=-1)[0]
        else:
            raise ValueError('unknown aggregation function {}'.format(aggregation_mode))

        self.output_dim = self.embedding_dim
        if aggregation_mode == "concatenate":
            self.output_dim *= self.num_hashes
        if self.append_weight:
            self.output_dim += self.num_hashes

        self.reset_parameters()

    def reset_parameters(self,
                         init_shared=lambda x: normal(x, std=0.1),
                         init_importance=lambda x: normal(x, std=0.0005)):
        """Resets the trainable parameters."""
        def set_constant_row(parameters, iRow=0, value=0):
            """Return `parameters` with row `iRow` as s constant `value`."""
            data = parameters.data
            data[iRow, :] = value
            return torch.nn.Parameter(data, requires_grad=parameters.requires_grad)

        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

    def forward(self, input):
        idx_importance_weights = input % self.num_embeddings
        # THERE IS NO ADVANTAGE OF USING THE FOLLWOING LINE, I JUST HAVE TO COMPARE WITH THE ALGORITHM IN THE PAPER
        input = idx_importance_weights if self.oldAlgorithm else input
        idx_shared_embeddings = torch.stack([h(input).masked_fill_(input == 0, 0) for h in self.hashes], dim=-1)

        shared_embedding = torch.stack([self.shared_embeddings(idx_shared_embeddings[:, :, iHash])
                                        for iHash in range(self.num_hashes)], dim=-1)
        importance_weight = self.importance_weights(idx_importance_weights)
        importance_weight = importance_weight.unsqueeze(-2)
        word_embedding = self.aggregate(importance_weight * shared_embedding)
        if self.append_weight:
            # concateates the vector with the weights
            word_embedding = torch.cat([word_embedding, importance_weight.squeeze(-2)], dim=-1)
        return word_embedding
    
    def get_hash_embedding_indices(self, input):
        if input.dim() == 1:
            input = torch.unsqueeze(input, 0)
        res = (((self.multiplers_tensor * input + self.adders_tensor) % self.moduler) % self.bins)
        idx_shared_embeddings = res.permute(1, 2, 0)
        r, c = idx_shared_embeddings.shape[1], idx_shared_embeddings.shape[2]
        idx_shared_embeddings = idx_shared_embeddings.reshape(idx_shared_embeddings.shape[0], -1)
        offsets = torch.arange(0, r * c, c)
        offsets = offsets.repeat(idx_shared_embeddings.shape[0], 1)
        if idx_shared_embeddings.shape[0] == 1:
            idx_shared_embeddings = torch.squeeze(idx_shared_embeddings, 0)
        if offsets.shape[0] == 1:
            offsets = torch.squeeze(offsets, 0)
        return idx_shared_embeddings, offsets

    def update_k(self, new_k: int) -> None:
        """Regenerate hash multipliers/adders only; keep embedding table (sim-hash k updates)."""
        dev = self.multiplers_tensor.device
        new_family = HashFamily(self.bins, moduler=self.moduler)
        new_mult, new_add = new_family.draw_hashes_tensor(new_k, device=str(dev))
        self.multiplers_tensor = new_mult
        self.adders_tensor = new_add
        self.num_hashes = new_k
        self.hashes = new_family.draw_hashes(new_k)
    
    def get_hash_embedding_tensors(self, input):
        """
        input: 1d tensor n
        res: 2d tensor k x n
        idx_shared_embeddings: 2d tensor k x n
        """
        if input.dim() == 1:
            input = torch.unsqueeze(input, 0)
        # Avoid modulo-by-zero when bins is small, and keep indices within this segment.
        denom = max(self.bins - 1, 1)
        res = ((self.multiplers_tensor * input + self.adders_tensor) % self.moduler) % denom + 1 + self.short_head_offsets
        # Valid range is [1 + offset, (bins - 1) + offset] when bins >= 2.
        # Clamp defensively to prevent OOB indices reaching EmbeddingBag.
        upper = max(self.short_head_offsets + self.bins - 1, self.short_head_offsets)
        res = res.clamp(self.short_head_offsets, upper)
        idx_shared_embeddings = res.squeeze(1)
        return idx_shared_embeddings


def profile_hash_embedding_collisions_impl(hashed_indices):
    hashed_indices = hashed_indices.T
    m, n = hashed_indices.shape[0], hashed_indices.shape[1]
    num_representation_collisions = 0
    num_digit_collisions = 0
    for i in range(m):
        for j in range(i + 1, m):
            if torch.equal(hashed_indices[i], hashed_indices[j]):
                num_representation_collisions += 1
            comp = hashed_indices[i] == hashed_indices[j]
            num_digit_collisions += torch.sum(comp).item()
    print(f"num_representation_collisions: {num_representation_collisions}")
    print(f"num_digit_collisions: {num_digit_collisions}")
    num_comparisons = m * (m - 1) // 2 * n
    num_non_collisions = num_comparisons - num_digit_collisions
    print(f"num_non_collisions: {num_non_collisions}")
    print(f"num_comparisons: {num_comparisons}")
    print(f"representation_collisions rate: {num_representation_collisions / (num_comparisons / n) * 100} %")
    print(f"digit_collisions rate: {num_digit_collisions / num_comparisons * 100} %")
    print(f"digit_non_collisions rate: {num_non_collisions / num_comparisons * 100} %")


def profile_hash_embedding_collisions(table_size, hash_embedding):
    print(f"hash capacity: {hash_embedding.num_buckets - 1} num_embeddings: {hash_embedding.num_embeddings} num_hashes: {hash_embedding.num_hashes}")
    indices = torch.arange(table_size)
    hashed_indices = hash_embedding.get_hash_embedding_tensors(indices)
    print(f"table size: {table_size}")
    profile_hash_embedding_collisions_impl(hashed_indices)


def test_profile_hash_embedding_collisions_impl():
    hashed_indices = torch.tensor([[2, 1, 3, 2, 1], [4, 2, 4, 1, 2]])
    profile_hash_embedding_collisions_impl(hashed_indices)


def test_profile_hash_embedding_collisions():
    capacity = 10
    num_hashes = 2
    table_size = 500
    hash_embedding = HashEmbedding(0, -1, num_buckets=capacity, num_hashes=num_hashes, append_weight=False)
    profile_hash_embedding_collisions(table_size, hash_embedding)


def test_get_hash_embedding_tensors_input_1d():
    embedding = HashEmbedding(1000000, 16, num_buckets=5, num_hashes=3, append_weight=False)
    input = torch.tensor([1,2,4,5,6])
    print(f"input: {input}")
    print(f"input.shape: {input.shape}")
    idx_shared_embeddings = embedding.get_hash_embedding_tensors(input)
    print(f"idx_shared_embeddings.shape: {idx_shared_embeddings.shape}")
    print(f"idx_shared_embeddings: {idx_shared_embeddings}")

def test_get_hash_embedding_tensors_input_2d():
    embedding = HashEmbedding(1000000, 16, num_buckets=5, num_hashes=3, append_weight=False)
    input = torch.tensor([[1,2,4,5,6], [1,2,2,3,4]])
    print(f"input: {input}")
    print(f"input.shape: {input.shape}")
    idx_shared_embeddings = embedding.get_hash_embedding_tensors(input)
    print(f"idx_shared_embeddings.shape: {idx_shared_embeddings.shape}")
    print(f"idx_shared_embeddings: {idx_shared_embeddings}")


if __name__ == "__main__":
    # test_get_hash_embedding_tensors_input_1d()
    # test_get_hash_embedding_tensors_input_2d()
    # test_profile_hash_embedding_collisions_impl()
    test_profile_hash_embedding_collisions()
