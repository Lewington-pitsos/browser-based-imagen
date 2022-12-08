import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
import random
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

unet = Unet(
    dim = 128, # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
    cond_dim = 256,
    dim_mults = (1, 2, 4), # the channel dimensions inside the model (multiplied by dim)
    num_resnet_blocks = 3,
    layer_attns = (False, True, True),
    layer_cross_attns = (False, True, True)
)

# imagen = Imagen(
#     unets = unet,
#     image_sizes = 32,
#     timesteps = 1000,
#     cond_drop_prob = 0.1
# ).cuda()

# # mock images (get a lot of this) and text encodings from large T5

# # text_embeds = torch.randn(4, 32, 768).cuda()
# # images = torch.randn(4, 3, 32, 32).cuda()


# images = imagen.sample(texts = [
#     'a whale breaching from afar',
#     # 'young girl blowing out candles on her birthday cake',
#     # 'fireworks with blue and green sparkles'
# ], cond_scale = 3)

# images[0].save("lol.png")


torch.onnx.export(unet, [(1, 3, 32, 32), (0.5,)], "unet.onnx", input_names=['image'], output_names=['prediction'])