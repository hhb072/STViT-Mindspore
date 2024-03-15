# --------------------------------------------------------
# Super Token Vision Transformer (STViT)
# Copyright (c) 2023 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as weight_init_
from mindspore import Parameter, Tensor

import math
import numpy as np
import collections.abc
from itertools import repeat

def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor = ops.uniform(tensor.shape, Tensor(2 * l - 1, mindspore.float32), Tensor(2 * u - 1, mindspore.float32), dtype=mindspore.float32)

    tensor = ops.erfinv(tensor)

    tensor = ops.mul(tensor, std * math.sqrt(2.))

    tensor = ops.add(tensor, mean)

    tensor = ops.clamp(tensor, Tensor(a, mindspore.float32), Tensor(b, mindspore.float32))

    return tensor

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Cell):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# class SwishImplementation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * ops.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_tensors[0]
#         sigmoid_i = ops.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Cell):
    def construct(self, x):
        return x * ops.sigmoid(x)

class LayerNorm2d(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(tuple([dim]), epsilon=1e-6)
        
    def construct(self, x):
        # return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous() 
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  

class ResDWC(nn.Cell):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding=kernel_size//2, has_bias=True, pad_mode='pad', group=dim)       
        
    def construct(self, x):
        return x + self.conv(x)

class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, has_bias=True)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, has_bias=True)
        self.drop = nn.Dropout(p=drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def construct(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
 
class Attention(nn.Cell):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, has_bias=True)
        self.proj_drop = nn.Dropout(p=proj_drop)
        

    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, 2) # (B, num_heads, head_dim, N)
        
        attn = (k.swapaxes(-1, -2) @ q) * self.scale
        
        attn = nn.Softmax(-2)(attn) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Cell):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = ops.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = Parameter(Tensor(weights), requires_grad=False)
           
        
    def construct(self, x):
        b, c, h, w = x.shape
        x = ops.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2, pad_mode = 'pad')        
        return x.reshape(b, c*9, h*w)

# class Fold():
#     def __init__(self, kernel_size=3):
#         self.kernel_size = kernel_size           
        
#     def call(self, x):
#         b, _, h, w = x.shape
#         x = nn.Conv2dTranspose(in_channels=self.kernel_size**2,
#                         out_channels=1,
#                         kernel_size=self.kernel_size,
#                         stride=1,
#                         padding=self.kernel_size//2,
#                         pad_mode="pad",
#                         has_bias=True)(x)
#         return x

class StokenAttention(nn.Cell):##############################################################################################################################################3333
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5

        self.kernel_size = 3
        
        self.unfold = Unfold(self.kernel_size)
        self.fold = nn.Conv2dTranspose(in_channels=self.kernel_size**2,
                        out_channels=1,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size//2,
                        pad_mode="pad",
                        has_bias=True)
        
        if refine:
            
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.SequentialCell(
                    nn.Conv2d(dim, dim, 1, 1, 0, has_bias=True),
                    nn.Conv2d(dim, dim, 5, 1, padding=2, has_bias=True, pad_mode='pad', group=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0, has_bias=True)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = ops.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        # 976
        
        stoken_features = ops.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        # 955
        
        # 935
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        # 911
        
        # with torch.no_grad():
        for idx in range(self.n_iter):
            stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
            stoken_features = stoken_features.swapaxes(1, 2).reshape(B, hh*ww, C, 9)
            affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
            # 874
            affinity_matrix = nn.Softmax(-1)(affinity_matrix) # (B, hh*ww, h*w, 9)
            # 871
            affinity_matrix_sum = affinity_matrix.sum(2).swapaxes(1, 2).reshape(B, 9, hh, ww)
            # 777
            affinity_matrix_sum = self.fold(affinity_matrix_sum)            

            if idx < self.n_iter - 1:
                stoken_features = pixel_features.swapaxes(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                # 853
                stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                # 777            
            
                # 771
                stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                # 767
            else:
                stoken_features = pixel_features.swapaxes(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)

                # 853
                stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                
                
                stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                #stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
                # 767
                
                if self.refine:
                    stoken_features = self.stoken_refine(stoken_features)
                    
                # 727
                
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.swapaxes(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
                # 714
                pixel_features = stoken_features @ affinity_matrix.swapaxes(-1, -2) # (B, hh*ww, C, h*w)
                # 687
                pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def construct(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class StokenAttentionLayer(nn.Cell):
    def __init__(self, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = Parameter(Tensor(init_values * ops.ones([1, dim, 1, 1])),requires_grad=True)
            self.gamma_2 = Parameter(Tensor(init_values * ops.ones([1, dim, 1, 1])),requires_grad=True)
        
    def construct(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x

class BasicLayer(nn.Cell):        
    def __init__(self, num_layers, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5,
                 downsample=False,
                 use_checkpoint=False, checkpoint_num=None):
        super().__init__()        
                
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
                
        self.blocks = nn.CellList([StokenAttentionLayer(
                                           dim=dim[0],  n_iter=n_iter, stoken_size=stoken_size,                                           
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           act_layer=act_layer, 
                                           layerscale=layerscale, init_values=init_values) for i in range(num_layers)])
                                           
                                                                           
                
        if downsample:            
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None
         
    def construct(self, x):
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
       
class PatchEmbed(nn.Cell):        
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=1, has_bias=True, pad_mode='pad'),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, padding=1, has_bias=True, pad_mode='pad'),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
                        
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, has_bias=True, pad_mode='pad'),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),            
            
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, has_bias=True, pad_mode='pad'),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            
        )

    def construct(self, x):
        x = self.proj(x)
        return x

class PatchMerging(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
        )

    def construct(self, x):
        x = self.proj(x)
        return x

class STViT(nn.Cell):   
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 n_iter=[1, 1, 1, 1], stoken_size=[8, 4, 2, 1],                
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 use_checkpoint=False, checkpoint_num=[0,0,0,0], 
                 layerscale=[False, False, False, False], init_values=1e-6, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim        
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        self.freeze_bn = freeze_bn

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [float(x) for x in ops.linspace(Tensor(0, mindspore.float32), Tensor(drop_path_rate, mindspore.float32), Tensor(sum(depths), mindspore.int32))]  # stochastic depth decay rule


        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],                              
                               n_iter=n_iter[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),                                                       
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,
                               use_checkpoint=use_checkpoint,
                               checkpoint_num=checkpoint_num[i_layer],                               
                               layerscale=layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer) 
    
        self.proj = nn.Conv2d(self.num_features, projection, 1, has_bias=True) if projection else None
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Dense(projection or self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()
        
    def init_weights(self):
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(weight_init_.initializer(
                        weight_init_.TruncatedNormal(sigma=0.02),
                        m.weight.shape,
                        m.weight.dtype))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    m.bias.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                                    m.bias.shape,
                                                                    m.bias.dtype))
            elif isinstance(m, nn.LayerNorm):
                m.gamma.set_data(weight_init_.initializer(weight_init_.One(),
                                                                m.gamma.shape,
                                                                m.gamma.dtype))
                m.beta.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                                m.beta.shape,
                                                                m.beta.dtype))

    def construct(self, x):
        x = self.patch_embed(x)     
        x = self.pos_drop(x) 
        for layer in self.layers:         
            x = layer(x)
            
        x = self.proj(x)
        x = self.norm(x)
        x = self.swish(x)
        x = self.avgpool(x)
        x = ops.flatten(x) # B C 1     
        x = self.head(x)
        return x
 
def stvit_small():
    model = STViT(embed_dim=[64, 128, 320, 512], # 25M, 4.4G, 677FPS
                    depths=[3, 5, 9, 3],
                    num_heads=[1, 2, 5, 8],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,                    
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    drop_path_rate=0.1, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, False, False],
                    init_values=1e-5,)
    # model.default_cfg = _cfg()
    return model    

def stvit_base():
    model = STViT(embed_dim=[96, 192, 384, 512], # 52M, 9.9G, 361 FPS
                    depths=[4, 6, 14, 6],
                    num_heads=[2, 3, 6, 8],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,                   
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    drop_path_rate=0.1,  
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, True, True],
                    init_values=1e-6,)
    model.default_cfg = _cfg()
    return model   

def stvit_large():
    model = STViT(embed_dim=[96, 192, 448, 640], # 95M, 15.6G, 269 FPS
                    depths=[4, 7, 19, 8],
                    num_heads=[2, 3, 7, 10],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1], # for 224/384
                    # stoken_size= [16, 8, 1, 1],
                    projection=1024,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    drop_path_rate=0.1, 
                    use_checkpoint=False,
                    checkpoint_num=[4,7,15,0],
                    layerscale=[False, False, True, True],
                    init_values=1e-6,)
    model.default_cfg = _cfg()
    return model    
    
def test():
    model =  STViT(                    
                    embed_dim=[96, 192, 448, 640], # 95M, 15.6G, 269 FPS
                    depths=[4, 7, 19, 8],
                    num_heads=[2, 3, 7, 10],
                   
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[16, 8, 2, 1],
                    
                    projection=1024,
                    
                    mlp_ratio=4,
                    stoken_refine=True,
                    stoken_refine_attention=True,
                    hard_label=False,
                    rpe=False,                    
                    qkv_bias=True,
                    qk_scale=None,
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False]*4,
                    init_values=1e-6,)
                
    print(model)
       
if __name__ == '__main__':
    test()    

