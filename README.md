# Browser Based Imagen

![Demo](demo/demo.gif)

This projects contains the scripts required for running imagen in the browser (i.e. using the actual chrome runtime). The GIF is sped up around 1000x, so in reality it  isn't practical in any way, but hopefully this repo will be useful to someone else trying to put text2img models in the browser. 

## How irt Works

Imagen works by taking a text prompt, running it through a t5 tokenizer, and then a t5 encoder (a.k.a "transformer"), before passing the encoded text plus some random noise to a unet, and performing a long sampling loop.

![How Imagen Works](demo/model.png)

This is the process I followed:

1. Trained a very small imagen model on cifar-10 classes using google colab (see links)
2. Converted the unet portion into onnx format using [torch.onnx](https://pytorch.org/docs/stable/onnx.html) (see `export_unet.ipynb`)
3. Downloaded an existing t5 transformer/tokenizer combo and exported these to ONNX/json (see `export_t5.ipynb`) respectively
4. Loaded all 3 (tokenizer, transformer, unet) into a chrome extension and used them to create a 250-step javascript inference loop where the output at each step is projected onto a html canvas

## Links
The actual extension can be found on [huggingface](https://huggingface.co/lewington/browser-based-imagen/tree/main). Simply download, unzip locally, and then load the `build` directory into chrome using "load unpacked"

The script for training a tiny imagen model can be found in [this colab](https://colab.research.google.com/drive/1QZ6Gys5dYnojn4_fnn3aPkNRZaifHODt?usp=sharing). 

The wanb report from some successful training runs can be found [here](https://wandb.ai/lewington/cifar10-imagen/reports/Cifar-10-Imagen-Training--VmlldzozMDU5MzEw?accessToken=hs40l4pznlt11xxlj58ch6xms40jcp1zckg4n5cyv0zs2q35vkf3p2qm1sq0kvzg).
