import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset

# unet for imagen

unet = Unet(
    dim = 128, # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
    cond_dim = 256,
    dim_mults = (1, 2, 4), # the channel dimensions inside the model (multiplied by dim)
    num_resnet_blocks = 3,
    layer_attns = (False, True, True),
    layer_cross_attns = (False, True, True)
)

imagen = Imagen(
    unets = unet,
    image_sizes = 32,
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 32, 768).cuda()
images = torch.randn(4, 3, 32, 32).cuda()

# feed images into imagen, training each unet in the cascade

# loss = imagen(images, text_embeds = text_embeds, unet_number = 1)
# loss.backward()

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

print("finished training")

images = imagen.sample(texts = [
    'a whale breaching from afar',
    # 'young girl blowing out candles on her birthday cake',
    # 'fireworks with blue and green sparkles'
], cond_scale = 3., device='cpu')

print(images.shape) # (3, 3, 256, 256)