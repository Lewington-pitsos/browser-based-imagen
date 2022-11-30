# Plan

We aim to make a super lightweight diffusion model which can produce nice looking emoji, with a single batch of 8 taking less than 20 seconds.

## Model

We will use paella, because it is the fastest by a wide margin at the moment


### Speeds
```
seconds            [162, 43, 13]
latent resoulution [32,  16,  8]

```

basically: we have achieved our goal with a latent resolution of 8 

### params in unet: 
69,416,100 for dim of 64

252,711,076 for (reccomended) dim of 128



## Data

Training Imagen took:

### Unconditioned Landscapes:

- 4000 images
- batch size 12
- img size: 64*64
- epochs 500
- lr 3e-4

Looks kind of reasonable

### Cifar 10 Classes

- 60,000 images
- batch size 14
- 64x64
- epochs 300
- lr 3e-4
- 10 classes

### Toy Example

- 1 class, unconditioned
    - laughing

- 5,000 images
- resize to 32x32

6 seconds per image on my laptop so 8.3 hours for 5,000 images, very doable


### Our Proposed

- 6 classes
    - laugh
    - happy
    - angry
    - confused
    - sad
    - winking
- 60,000 images


### Citations: 

Cifar10
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}


