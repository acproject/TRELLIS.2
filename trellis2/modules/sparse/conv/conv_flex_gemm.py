import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d


_orig_init_hashmap = None


def _patched_init_hashmap(spatial_size, hashmap_size, device, with_values=True):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D

    if VOL < 2**32:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    else:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)

    if with_values:
        hashmap_vals = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)
        return hashmap_keys, hashmap_vals
    return hashmap_keys


def _ensure_hashmap_patched():
    global _orig_init_hashmap
    if _orig_init_hashmap is not None:
        return
    import flex_gemm.ops.utils as utils
    _orig_init_hashmap = utils.init_hashmap
    utils.init_hashmap = _patched_init_hashmap


_ensure_hashmap_patched()


def _chunked_subm_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    import flex_gemm.ops.spconv as spconv_ops
    import flex_gemm.ops.utils as utils
    from flex_gemm import kernels

    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw
    N = x.feats.shape[0]
    device = x.feats.device
    dtype = x.feats.dtype

    shape = torch.Size([*x.shape, *x.spatial_shape])
    Ns, Cs, W, H, D = shape

    hashmap_keys, hashmap_vals = utils.init_hashmap(
        shape, int(spconv_ops.HASHMAP_RATIO * x.coords.shape[0]), device
    )

    neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map(
        hashmap_keys, hashmap_vals, x.coords,
        W, H, D,
        Kw, Kh, Kd,
        self.dilation[0], self.dilation[1], self.dilation[2],
    )

    CHUNK = 8192
    weight_flat = self.weight.reshape(Co, V * Ci).transpose(0, 1)

    if N <= CHUNK:
        im2col = torch.zeros((N * V, Ci), device=device, dtype=dtype)
        mask = neighbor_map.view(-1) != 0xFFFFFFFF
        valid_indices = neighbor_map.view(-1).long()
        im2col[mask] = x.feats[valid_indices[mask]]
        im2col = im2col.view(N, V * Ci)

        if self.bias is not None:
            output = torch.addmm(self.bias, im2col, weight_flat)
        else:
            output = torch.mm(im2col, weight_flat)
    else:
        output = x.feats.new_empty(N, Co)
        if self.bias is not None:
            output[:] = self.bias.unsqueeze(0)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            chunk_size = end - start
            nm_chunk = neighbor_map[start:end]

            im2col = torch.zeros((chunk_size * V, Ci), device=device, dtype=dtype)
            mask = nm_chunk.view(-1) != 0xFFFFFFFF
            valid_indices = nm_chunk.view(-1).long()
            im2col[mask] = x.feats[valid_indices[mask]]
            im2col = im2col.view(chunk_size, V * Ci)

            if self.bias is not None:
                out_chunk = torch.addmm(self.bias, im2col, weight_flat)
            else:
                out_chunk = torch.mm(im2col, weight_flat)
            output[start:end] = out_chunk

    return x.replace(output)


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    if x.feats.numel() == 0:
        out_feats = x.feats.new_zeros(0, self.out_channels)
        return x.replace(out_feats)

    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw
    N = x.feats.shape[0]
    im2col_bytes = N * V * Ci * x.feats.element_size()
    available_bytes = 4 * 1024 * 1024 * 1024

    if im2col_bytes > available_bytes:
        return _chunked_subm_conv3d_forward(self, x)

    flex_gemm.ops.spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)
    
    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )
    
    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
    
    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
