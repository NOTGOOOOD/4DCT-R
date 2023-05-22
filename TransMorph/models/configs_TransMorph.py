'''
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import ml_collections
'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
rpe (bool): Enable relative position embedding. Default: True
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
'''
def get_3DTransMorph_config():
    '''
    Trainable params: 15,201,579
    '''

    """
       这段代码定义了一个名为config的ConfigDict对象，并设置了该对象的各种属性，以配置一个视觉转换器（Vision Transformer）的网络结构和超参数。
    """
    config = ml_collections.ConfigDict()
    config.if_transskip = True #一个布尔值，表示是否使用跳跃连接（skip connection）来加速网络训练
    config.if_convskip = True  #布尔值，是否在跳跃连接中使用卷积操作
    config.patch_size = 4   #输入图像被分成的小块的大小
    config.in_chans = 2  #输入图像的通道数
    config.embed_dim = 96     #嵌入层的维度
    config.depths = (2, 2, 4, 2)     #一个元组，每个阶段中Transformer块block数量
    config.num_heads = (4, 4, 8, 8)  #一个元组，每个阶段中的每个transformer块中的多头注意力机制的数量
    config.window_size = (5, 6, 7)   #一个元组，每个阶段中的局部注意力机制窗口大小
    config.mlp_ratio = 4  #MLP的隐藏层维度与嵌入层维度之比
    config.pat_merg_rf = 4   #块内位置嵌入的感受野的大小
    config.qkv_bias = False  #是否在多头注意力机制中使用偏置项
    config.drop_rate = 0  #随机失活的概率
    config.drop_path_rate = 0.3  #随机失活路径的概率
    config.ape = False    #是否使用不同的位置嵌入方案（相对位置嵌入、绝对位置嵌入）
    config.spe = False
    config.rpe = True
    config.patch_norm = True  #是否在每个块输入前使用批归一化
    config.use_checkpoint = False   #是否使用检查点来减少显存占用
    config.out_indices = (0, 1, 2, 3)   #输出的阶段索引
    config.reg_head_chan = 16   #分类器中的通道数
    # config.img_size = (160, 192, 224)  #输入图像的大小
    config.img_size = (160, 160, 160)  # 输入图像的大小
    return config

def get_3DTransMorphNoRelativePosEmbd_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphSin_config():
    '''
    TransMorph with Sinusoidal Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.pos_embed_method = 'relative'
    return config

def get_3DTransMorphLrn_config():
    '''
    TransMorph with Learnable Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = True
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphNoConvSkip_config():
    '''
    No skip connections from convolution layers

    Computational complexity:       577.34 GMac
    Number of parameters:           63.56 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.pos_embed_method = 'relative'
    return config

def get_3DTransMorphNoTransSkip_config():
    '''
    No skip connections from Transformer blocks

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphNoSkip_config():
    '''
    No skip connections

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphLarge_config():
    '''
    A Large TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphSmall_config():
    '''
    A Small TransMorph Network
    '''


    """
    这段代码定义了一个名为config的ConfigDict对象，并设置了该对象的各种属性，以配置一个视觉转换器（Vision Transformer）的网络结构和超参数。
    """
    config = ml_collections.ConfigDict()
    config.if_transskip = True  #一个布尔值，表示是否使用跳跃连接（skip connection）来加速网络训练
    config.if_convskip = True  #布尔值，是否在跳跃连接中使用卷积操作
    config.patch_size = 4  #输入图像被分成的小块的大小
    config.in_chans = 2  #输入图像的通道数
    config.embed_dim = 48  #嵌入层的维度
    config.depths = (2, 2, 4, 2)  #一个元组，每个阶段中Transformer块block数量
    config.num_heads = (4, 4, 4, 4)  #一个元组，每个阶段中的每个transformer块中的多头注意力机制的数量
    config.window_size = (5, 6, 7)  #一个元组，每个阶段中的局部注意力机制窗口大小
    config.mlp_ratio = 4  #MLP的隐藏层维度与嵌入层维度之比
    config.pat_merg_rf = 4  #块内位置嵌入的感受野的大小
    config.qkv_bias = False  #是否在多头注意力机制中使用偏置项
    config.drop_rate = 0  #随机失活的概率
    config.drop_path_rate = 0.3  #随机失活路径的概率
    config.ape = False  #是否使用不同的位置嵌入方案（相对位置嵌入、绝对位置嵌入）
    config.spe = False
    config.rpe = True
    config.patch_norm = True  #是否在每个块输入前使用批归一化
    config.use_checkpoint = False  #是否使用检查点来减少显存占用
    config.out_indices = (0, 1, 2, 3)  #输出的阶段索引
    config.reg_head_chan = 16 #分类器中的通道数
    config.img_size = (160, 192, 224)  #输入图像的大小
    return config

def get_3DTransMorphTiny_config():
    '''
    A Tiny TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 6
    config.depths = (2, 2, 4, 2)
    config.num_heads = (2, 2, 4, 4)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config
