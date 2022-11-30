import torch
from imagen_pytorch import Unet, Imagen
from torchviz import make_dot
from torchsummary import summary

# unet for imagen


unet = Unet(
    dim = 128, # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
    cond_dim = 256,
    dim_mults = (1, 2, 4),
    num_resnet_blocks = 2,
    layer_attns = (False, True, True),
    layer_cross_attns = (False, True, True)
)


y = unet(torch.randn(1, 3, 32, 32), torch.randn(2))

make_dot(y.mean(), params=dict(unet.named_parameters()), show_attrs=True, show_saved=True)

dot = make_dot(y)

dot.format = 'png'
dot.render('unet.png')


# summary(unet,  (3, 32, 32), device='cpu')