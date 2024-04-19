img_size = (3, 32, 32)

# Model config
chw = img_size[0] * img_size[1] * img_size[2]
d_model = 256
dff = 512
d_latent = 256
nlayers_decoder = 7
nheads_encoder = 8
nblocks_encoder = 5
dropout = 0.1
nbits = 8
