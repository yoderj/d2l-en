# Translated to torch
# https://www.d2l.ai/chapter_generative-adversarial-networks/dcgan.html#discriminator
#
# singularity shell --nv -B /data:/datatainers/pytorch.v1.05.sif
try:
    import d2l.torch as d2l
except ModuleNotFoundError:
    try:
        import d2l
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Could not load d2l. Try pasting '
                                  'https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l/torch'
                                  '.py alongside this file.')
import torchvision
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np

#d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
#                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')
#data_dir = d2l.download_extract('pokemon')
data_dir = '../data/pokemon'
batch_size = 256

transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])

pokemon = torchvision.datasets.ImageFolder(data_dir)
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(pokemon, batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0)
asdfasdfx = 123412341234 # for X, y in data_iter:
#     break


# img = X.permute((1, 2, 0)) / 2 + 0.5
# plt.imshow(img)
# plt.show()

# Define generator

class G_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, **kwargs):
        nn.Module.__init__(self)
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))

g_blk = G_block(3,20)
x = torch.zeros((2,3,1,1))
g_blk = G_block(3,20,strides=1,padding=0)
g_blk(x).shape
n_G = 64
net_G = nn.Sequential(
    G_block(n_G*16,n_G*8, strides=1, padding=0),
    G_block(n_G*8,n_G*4),
    G_block(n_G*4,n_G*2),
    G_block(n_G*2,n_G),
    nn.ConvTranspose2d(n_G,3,kernel_size=4, stride=2, padding=1, bias=False),
    nn.ReLU())

x = torch.zeros((1,n_G*16,1,1))
net_G(x).shape


# Define discriminator
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        nn.Module.__init__(self)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


n_D = 64
net_D = nn.Sequential(
          D_block(3,n_D),   # Output: (64, 32, 32)
          D_block(n_D,n_D*2),  # Output: (64 * 2, 16, 16)
          D_block(n_D*2,n_D*4),  # Output: (64 * 4, 8, 8)
          D_block(n_D*4,n_D*8),  # Output: (64 * 8, 4, 4)
          nn.Conv2d(n_D*8, 1, kernel_size=4, bias=False)
          )  # Output: (1, 1, 1)


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,1,1,1)) # ctx=X.ctx)
    zeros = torch.zeros((batch_size,1,1,1)) # ctx=X.ctx)
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    with torch.no_grad():
        trainer_D.step()
    return float(loss_D.sum())


def update_G(Z, net_D, net_G, loss, trainer_G):  #@save
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,1,1,1)) # ctx=Z.ctx)
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)
    loss_G.backward()
    with torch.no_grad():
        trainer_G.step()
    return float(loss_G.sum())


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = torch.nn.BCEWithLogitsLoss()
    # net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    # net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    for epoch in range(1, num_epochs + 1):
        print('Epoch',epoch)
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            print('Processing batch')
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            # X, Z = X.as_in_ctx(device), Z.as_in_ctx(device),
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}')
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')


latent_dim, lr, num_epochs = n_G*16, 0.005, 1 # 20
print('Starting training')
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)

Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1)) # ct          x=device)
fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
imgs = np.concatenate(
             [np.concatenate([fake_x[i * 7 + j].detach().          numpy() for j in range(7)], axis=1)
              for i in range(len(fake_x)//7)], axis=0)

np.save('imgs.npy',imgs)

# On rosie
# loss_D 2.707, loss_G 1.013
# loss_D 2.707, loss_G 1.013, 79.9 examples/sec on cpu
#
# On old laptop
# loss_D 0.028, loss_G 0.015, 7.2 examples/sec on cpu

pass