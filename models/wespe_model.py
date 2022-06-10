import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import numpy as np
from torch.autograd import grad
from torchvision.models import vgg19
from scipy import signal


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()

        self.model = vgg19(pretrained=True).features[:-1]
        for p in self.model.parameters():
            p.requires_grad = False

        # normalization
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.mean = nn.Parameter(data=mean, requires_grad=False)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.std = nn.Parameter(data=std, requires_grad=False)

    def forward(self, x, y):
        """
        Arguments:
            x, y: float tensors with shape [b, 3, h, w].
            They represent RGB images with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        b = x.size(0)
        x = torch.cat([x, y], dim=0)
        x = (x - self.mean)/self.std

        x = self.model(x)
        # relu_5_4 features,
        # a float tensor with shape [2 * b, 512, h/16, w/16]

        x, y = torch.split(x, b, dim=0)
        b, c, h, w = x.size()
        normalizer = b * c * h * w
        return ((x - y)**2).sum()/normalizer


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        b, c, h, w = x.size()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :(h - 1), :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :(w - 1)]), 2).sum()
        return (h_tv + w_tv)/(b * c * h * w)


def gradient_penalty(x, y, f):
    """
    Arguments:
        x, y: float tensors with shape [b, c, h, w].
        f: a pytorch module.
    Returns:
        a float tensor with shape [].
    """

    # interpolation
    b = x.size(0)
    alpha = torch.rand([b, 1, 1, 1]).to(x.device)
    z = x + alpha * (y - x)
    z.requires_grad = True

    # compute gradient
    ones = torch.ones(z.size(0)).to(z.device)
    g = grad(f(z), z, grad_outputs=ones, create_graph=True, only_inputs=True, retain_graph=True)[0]
    # it has shape [b, c, h, w]

    g = g.view(b, -1)
    return ((g.norm(p=2, dim=1) - 1.0)**2).mean(0)


class GaussianBlur(nn.Module):

    def __init__(self, chans=3):
        super(GaussianBlur, self).__init__()
        self.chans = chans

        def get_kernel(size=21, std=3):
            """Returns a 2D Gaussian kernel array."""
            k = signal.gaussian(size, std=std).reshape(size, 1)
            k = np.outer(k, k)
            return k/k.sum()

        kernel = get_kernel(size=11, std=3)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0).repeat([self.chans, 1, 1, 1])
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)  # shape [3, 1, 11, 11]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
        """
        x = F.conv2d(x, self.kernel, padding=5, groups=self.chans)
        return x


class Grayscale(nn.Module):

    def __init__(self):
        super(Grayscale, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 1, h, w].
        """
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return result.unsqueeze(1)


class WESPEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(lr=1e-4)
        parser.set_defaults(batch_size=64)
        parser.set_defaults(netD='wespe')
        parser.set_defaults(netG='wespe')
        parser.set_defaults(init_type='kaiming')
        if is_train:
            parser.add_argument('--lr_disc', type=float, default=4e-4, help='initial learning rate for discriminators.')
            parser.add_argument('--lambda_tv', type=float, default=10.0, help='weight of tv loss.')
            parser.add_argument('--lambda_ct', type=float,  default=5e-3, help='weight of texture and colour loss.')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'D_t', 'D_c', 'G_t', 'G_c', 'content', 'tv']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'rec_A']        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_G', 'G_F', 'D_c', 'D_t']
        else:  # during test time, only load Gs
            self.model_names = ['G_G', 'G_F']

        self.netG_G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.q, opt.is_residual, not opt.no_bias, opt.gen_kernel_sizes)    # Source to target domain
        self.netG_F = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.q, opt.is_residual, not opt.no_bias, opt.gen_kernel_sizes)    # Reconstruction

        if self.isTrain:  # define discriminators
            self.netD_c = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.q, opt.is_residual, not opt.no_bias, opt.disc_kernel_sizes)
            self.netD_t = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.q, opt.is_residual, not opt.no_bias, opt.disc_kernel_sizes)

        self.content_criterion = ContentLoss().to(self.device)
        self.tv_criterion = TVLoss().to(self.device)
        self.color_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.texture_criterion = nn.BCEWithLogitsLoss().to(self.device)

        betas = (opt.beta1, 0.999)
        self.G_optimizer = optim.Adam(lr=opt.lr, params=self.netG_G.parameters(), betas=betas)
        self.F_optimizer = optim.Adam(lr=opt.lr, params=self.netG_F.parameters(), betas=betas)
        self.c_optimizer = optim.Adam(lr=opt.lr_disc, params=self.netD_c.parameters(), betas=betas)
        self.t_optimizer = optim.Adam(lr=opt.lr_disc, params=self.netD_t.parameters(), betas=betas)

        self.optimizers.append(self.G_optimizer)
        self.optimizers.append(self.F_optimizer)
        self.optimizers.append(self.c_optimizer)
        self.optimizers.append(self.t_optimizer)

        self.blur = GaussianBlur(chans=opt.output_nc).to(self.device)
        self.gray = Grayscale().to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_G(self.real_A)  # G_A(A)
        self.rec_A = self.netG_F(self.fake_B)   # G_B(G_A(A))

    def backward_D(self):
        batch_size = self.real_A.size(0)
        pos_labels = torch.ones(batch_size, dtype=torch.float, device=self.real_A.device)
        neg_labels = torch.zeros(batch_size, dtype=torch.float, device=self.real_A.device)

        fake_B_blur = self.blur(self.fake_B)
        real_B_blur = self.blur(self.real_B)
        if self.fake_B.shape[1] == 1:   
            fake_B_gray = self.fake_B       # No change to already grayscale image
            real_B_gray = self.real_B       # No change to already grayscale image
        else:
            fake_B_gray = self.gray(self.fake_B)    # RGB to grayscale
            real_B_gray = self.gray(self.real_B)
        
        targets = torch.cat([pos_labels, neg_labels], dim=0)

        is_real_real = self.netD_c(real_B_blur)
        is_fake_real = self.netD_c(fake_B_blur.detach())
        logits = torch.cat([is_real_real, is_fake_real], dim=0)
        self.loss_D_c = self.color_criterion(logits, targets)

        is_real_real = self.netD_t(real_B_gray)
        is_fake_real = self.netD_t(fake_B_gray.detach())
        logits = torch.cat([is_real_real, is_fake_real], dim=0)
        self.loss_D_t = self.texture_criterion(logits, targets)

        self.loss_D = self.loss_D_c + self.loss_D_t
        self.c_optimizer.zero_grad()
        self.t_optimizer.zero_grad()
        self.loss_D.backward()
        self.c_optimizer.step()
        self.t_optimizer.step()


    def backward_G(self):
        """Calculate the loss for generators G_G and G_F"""
        
        self.loss_content = self.content_criterion(self.real_A, self.rec_A)
        self.loss_tv = self.tv_criterion(self.fake_B)

        batch_size = self.real_A.size(0)
        pos_labels = torch.ones(batch_size, dtype=torch.float, device=self.real_A.device)

        fake_B_blur = self.blur(self.fake_B)
        self.loss_G_c = self.color_criterion(self.netD_c(fake_B_blur), pos_labels)

        if self.fake_B.shape[1] == 1:   
            fake_B_gray = self.fake_B       # No change to already grayscale image
        else:
            fake_B_gray = self.gray(self.fake_B)    # RGB to grayscale

        self.loss_G_t = self.texture_criterion(self.netD_t(fake_B_gray), pos_labels)

        self.loss_G = self.loss_content + self.opt.lambda_tv * self.loss_tv
        self.loss_G += self.opt.lambda_ct * (self.loss_G_c + self.loss_G_t)

        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()
        self.loss_G.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()
        

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # generators G_G and G_F
        self.set_requires_grad([self.netD_c, self.netD_t], False)  # Ds require no gradients when optimizing Gs
        self.backward_G()             # calculate gradients for generators G_G and G_F
        # netD_t and netD_c
        self.set_requires_grad([self.netD_c, self.netD_t], True)
        self.backward_D()
       