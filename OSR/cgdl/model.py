import numpy as np
import torch
import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False
nllloss = nn.NLLLoss()

class CONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding, stride, flat_dim, latent_dim):
        super(CONV, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.flat_dim = flat_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, stride=stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(out_ch, affine=False),
            nn.PReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )

    def encode(self, x):
        h = self.net(x)
        h_flat = h.view(-1, self.out_ch*self.flat_dim*self.flat_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class TCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, out_dim, t_latent_dim):
        super(TCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.out_dim = out_dim
        self.t_latent_dim = t_latent_dim

        self.fc = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.net = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch, affine=False),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )

    def decode(self, x):
        x = self.fc(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        h = self.net(x)
        h_flat = h.view(-1, self.t_out_ch * self.out_dim * self.out_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class FCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride):
        super(FCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding

        self.fc_final = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            nn.Tanh()
        )

    def final_decode(self,x):
        x = self.fc_final(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        x_re = self.final(x)
        return x_re

class LVAE(nn.Module):
    def __init__(self, in_ch=3,
                 out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                 kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, padding2=2, stride1=1, stride2=2,
                 flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                 latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=32, num_class =15):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch64 = out_ch64
        self.out_ch128 = out_ch128
        self.out_ch256 = out_ch256
        self.out_ch512 = out_ch512
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.padding0 = padding0
        self.padding1 = padding1
        self.padding2 = padding2
        self.stride1 = stride1
        self.stride2 = stride2
        self.flat_dim32 = flat_dim32
        self.flat_dim16 = flat_dim16
        self.flat_dim8 = flat_dim8
        self.flat_dim4 = flat_dim4
        self.flat_dim2 = flat_dim2
        self.flat_dim1 = flat_dim1
        self.latent_dim512 = latent_dim512
        self.latent_dim256 = latent_dim256
        self.latent_dim128 = latent_dim128
        self.latent_dim64 = latent_dim64
        self.latent_dim32 = latent_dim32
        self.num_class = num_class

        # initialize required CONVs
        self.CONV1_1 = CONV(self.in_ch, self.out_ch64, self.kernel1, self.padding2, self.stride1, self.flat_dim32,
                            self.latent_dim512)
        self.CONV1_2 = CONV(self.out_ch64, self.out_ch64, self.kernel3, self.padding1, self.stride2, self.flat_dim16,
                            self.latent_dim512)

        self.CONV2_1 = CONV(self.out_ch64, self.out_ch128, self.kernel3, self.padding1, self.stride1, self.flat_dim16, self.latent_dim256)
        self.CONV2_2 = CONV(self.out_ch128, self.out_ch128, self.kernel3, self.padding1, self.stride2, self.flat_dim8, self.latent_dim256)

        self.CONV3_1 = CONV(self.out_ch128, self.out_ch256, self.kernel3, self.padding1, self.stride1, self.flat_dim8,
                            self.latent_dim128)
        self.CONV3_2 = CONV(self.out_ch256, self.out_ch256, self.kernel3, self.padding1, self.stride2, self.flat_dim4,
                            self.latent_dim128)

        self.CONV4_1 = CONV(self.out_ch256, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim4,
                            self.latent_dim64)
        self.CONV4_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim2,
                            self.latent_dim64)

        self.CONV5_1 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim2,
                            self.latent_dim32)
        self.CONV5_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim1,
                            self.latent_dim32)

        # initialize required TCONVs
        self.TCONV5_2 = TCONV(self.latent_dim32, self.flat_dim1, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim2, self.latent_dim32)
        self.TCONV5_1 = TCONV(self.latent_dim32, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim2, self.latent_dim64)

        self.TCONV4_2 = TCONV(self.latent_dim64, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim4, self.latent_dim64)
        self.TCONV4_1 = TCONV(self.latent_dim64, self.flat_dim4, self.out_ch512, self.out_ch256, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim4, self.latent_dim128)

        self.TCONV3_2 = TCONV(self.latent_dim128, self.flat_dim4, self.out_ch256, self.out_ch256, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim8, self.latent_dim128)
        self.TCONV3_1 = TCONV(self.latent_dim128, self.flat_dim8, self.out_ch256, self.out_ch128, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim8, self.latent_dim256)

        self.TCONV2_2 = TCONV(self.latent_dim256, self.flat_dim8, self.out_ch128, self.out_ch128, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim16, self.latent_dim256)
        self.TCONV2_1 = TCONV(self.latent_dim256, self.flat_dim16, self.out_ch128, self.out_ch64, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim16, self.latent_dim512)

        self.TCONV1_2 = TCONV(self.latent_dim512, self.flat_dim16, self.out_ch64, self.out_ch64, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim32, self.latent_dim512)
        self.TCONV1_1 = FCONV(self.latent_dim512, self.flat_dim32, self.out_ch64, self.in_ch, self.kernel1,
                              self.padding2, self.stride1)

        self.classifier = nn.Linear(self.latent_dim32, self.num_class)
        self.one_hot = nn.Linear(self.num_class, self.latent_dim32)

    def lnet(self, x, y_de):
        # ---deterministic upward pass
        # upwards
        enc1_1, mu_up1_1, var_up1_1 = self.CONV1_1.encode(x)
        enc1_2, mu_up1_2, var_up1_2 = self.CONV1_2.encode(enc1_1)

        enc2_1, mu_up2_1, var_up2_1 = self.CONV2_1.encode(enc1_2)
        enc2_2, mu_up2_2, var_up2_2 = self.CONV2_2.encode(enc2_1)

        enc3_1, mu_up3_1, var_up3_1 = self.CONV3_1.encode(enc2_2)
        enc3_2, mu_up3_2, var_up3_2 = self.CONV3_2.encode(enc3_1)

        enc4_1, mu_up4_1, var_up4_1 = self.CONV4_1.encode(enc3_2)
        enc4_2, mu_up4_2, var_up4_2 = self.CONV4_2.encode(enc4_1)

        enc5_1, mu_up5_1, var_up5_1 = self.CONV5_1.encode(enc4_2)
        enc5_2, mu_latent, var_latent = self.CONV5_2.encode(enc5_1)

        latent = ut.sample_gaussian(mu_latent, var_latent)
        predict = F.log_softmax(self.classifier(latent), dim=1)
        predict_test = F.log_softmax(self.classifier(mu_latent), dim=1)
        yh = self.one_hot(y_de)

        # partially downwards
        dec5_1, mu_dn5_1, var_dn5_1 = self.TCONV5_2.decode(latent)
        prec_up5_1 = var_up5_1 ** (-1)
        prec_dn5_1 = var_dn5_1 ** (-1)
        qmu5_1 = (mu_up5_1 * prec_up5_1 + mu_dn5_1 * prec_dn5_1) / (prec_up5_1 + prec_dn5_1)
        qvar5_1 = (prec_up5_1 + prec_dn5_1) ** (-1)
        de_latent5_1 = ut.sample_gaussian(qmu5_1, qvar5_1)

        dec4_2, mu_dn4_2, var_dn4_2 = self.TCONV5_1.decode(de_latent5_1)
        prec_up4_2 = var_up4_2 ** (-1)
        prec_dn4_2 = var_dn4_2 ** (-1)
        qmu4_2 = (mu_up4_2 * prec_up4_2 + mu_dn4_2 * prec_dn4_2) / (prec_up4_2 + prec_dn4_2)
        qvar4_2 = (prec_up4_2 + prec_dn4_2) ** (-1)
        de_latent4_2 = ut.sample_gaussian(qmu4_2, qvar4_2)

        dec4_1, mu_dn4_1, var_dn4_1 = self.TCONV4_2.decode(de_latent4_2)
        prec_up4_1 = var_up4_1 ** (-1)
        prec_dn4_1 = var_dn4_1 ** (-1)
        qmu4_1 = (mu_up4_1 * prec_up4_1 + mu_dn4_1 * prec_dn4_1) / (prec_up4_1 + prec_dn4_1)
        qvar4_1 = (prec_up4_1 + prec_dn4_1) ** (-1)
        de_latent4_1 = ut.sample_gaussian(qmu4_1, qvar4_1)

        dec3_2, mu_dn3_2, var_dn3_2 = self.TCONV4_1.decode(de_latent4_1)
        prec_up3_2 = var_up3_2 ** (-1)
        prec_dn3_2 = var_dn3_2 ** (-1)
        qmu3_2 = (mu_up3_2 * prec_up3_2 + mu_dn3_2 * prec_dn3_2) / (prec_up3_2 + prec_dn3_2)
        qvar3_2 = (prec_up3_2 + prec_dn3_2) ** (-1)
        de_latent3_2 = ut.sample_gaussian(qmu3_2, qvar3_2)

        dec3_1, mu_dn3_1, var_dn3_1 = self.TCONV3_2.decode(de_latent3_2)
        prec_up3_1 = var_up3_1 ** (-1)
        prec_dn3_1 = var_dn3_1 ** (-1)
        qmu3_1 = (mu_up3_1 * prec_up3_1 + mu_dn3_1 * prec_dn3_1) / (prec_up3_1 + prec_dn3_1)
        qvar3_1 = (prec_up3_1 + prec_dn3_1) ** (-1)
        de_latent3_1 = ut.sample_gaussian(qmu3_1, qvar3_1)

        dec2_2, mu_dn2_2, var_dn2_2 = self.TCONV3_1.decode(de_latent3_1)
        prec_up2_2 = var_up2_2 ** (-1)
        prec_dn2_2 = var_dn2_2 ** (-1)
        qmu2_2 = (mu_up2_2 * prec_up2_2 + mu_dn2_2 * prec_dn2_2) / (prec_up2_2 + prec_dn2_2)
        qvar2_2 = (prec_up2_2 + prec_dn2_2) ** (-1)
        de_latent2_2 = ut.sample_gaussian(qmu2_2, qvar2_2)

        dec2_1, mu_dn2_1, var_dn2_1 = self.TCONV2_2.decode(de_latent2_2)
        prec_up2_1 = var_up2_1 ** (-1)
        prec_dn2_1 = var_dn2_1 ** (-1)
        qmu2_1 = (mu_up2_1 * prec_up2_1 + mu_dn2_1 * prec_dn2_1) / (prec_up2_1 + prec_dn2_1)
        qvar2_1 = (prec_up2_1 + prec_dn2_1) ** (-1)
        de_latent2_1 = ut.sample_gaussian(qmu2_1, qvar2_1)

        dec1_2, mu_dn1_2, var_dn1_2 = self.TCONV2_1.decode(de_latent2_1)
        prec_up1_2 = var_up1_2 ** (-1)
        prec_dn1_2 = var_dn1_2 ** (-1)
        qmu1_2 = (mu_up1_2 * prec_up1_2 + mu_dn1_2 * prec_dn1_2) / (prec_up1_2 + prec_dn1_2)
        qvar1_2 = (prec_up1_2 + prec_dn1_2) ** (-1)
        de_latent1_2 = ut.sample_gaussian(qmu1_2, qvar1_2)

        dec1_1, mu_dn1_1, var_dn1_1 = self.TCONV1_2.decode(de_latent1_2)
        prec_up1_1 = var_up1_1 ** (-1)
        prec_dn1_1 = var_dn1_1 ** (-1)
        qmu1_1 = (mu_up1_1 * prec_up1_1 + mu_dn1_1 * prec_dn1_1) / (prec_up1_1 + prec_dn1_1)
        qvar1_1 = (prec_up1_1 + prec_dn1_1) ** (-1)
        de_latent1_1 = ut.sample_gaussian(qmu1_1, qvar1_1)

        x_re = self.TCONV1_1.final_decode(de_latent1_1)

        return latent, mu_latent, var_latent, \
               qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
               qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
               predict, predict_test, yh, \
               x_re, \
               mu_dn5_1, var_dn5_1, mu_dn4_2, var_dn4_2, mu_dn4_1, var_dn4_1, mu_dn3_2, var_dn3_2, mu_dn3_1, var_dn3_1, \
               mu_dn2_2, var_dn2_2, mu_dn2_1, var_dn2_1, mu_dn1_2, var_dn1_2, mu_dn1_1, var_dn1_1

    def loss(self, x, y, y_de, beta, lamda):

        latent, mu_latent, var_latent, \
        qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        predict, predict_test, yh, \
        x_re, \
        pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de)

        rec = reconstruction_function(x_re, x)

        pm, pv = torch.zeros(mu_latent.shape).cuda(), torch.ones(var_latent.shape).cuda()
        # print("mu1", mu1)
        kl_latent = ut.kl_normal(mu_latent, var_latent, pm, pv, yh)
        kl5_1 = ut.kl_normal(qmu5_1, qvar5_1, pmu5_1, pvar5_1, 0)
        kl4_2 = ut.kl_normal(qmu4_2, qvar4_2, pmu4_2, pvar4_2, 0)
        kl4_1 = ut.kl_normal(qmu4_1, qvar4_1, pmu4_1, pvar4_1, 0)
        kl3_2 = ut.kl_normal(qmu3_2, qvar3_2, pmu3_2, pvar3_2, 0)
        kl3_1 = ut.kl_normal(qmu3_1, qvar3_1, pmu3_1, pvar3_1, 0)
        kl2_2 = ut.kl_normal(qmu2_2, qvar2_2, pmu2_2, pvar2_2, 0)
        kl2_1 = ut.kl_normal(qmu2_1, qvar2_1, pmu2_1, pvar2_1, 0)
        kl1_2 = ut.kl_normal(qmu1_2, qvar1_2, pmu1_2, pvar1_2, 0)
        kl1_1 = ut.kl_normal(qmu1_1, qvar1_1, pmu1_1, pvar1_1, 0)

        kl = beta * torch.mean(kl_latent + kl5_1 + kl4_2 + kl4_1 + kl3_2 + kl3_1 + kl2_2 + kl2_1 + kl1_2 + kl1_1)

        ce = nllloss(predict, y)

        nelbo = rec + kl + lamda*ce
        # nelbo = rec
        return nelbo, mu_latent, predict, predict_test, x_re,rec,kl,lamda*ce

    def test(self, x, y_de):
        _, mu_latent, _, \
        _, _, _, _, _, _, _, _, _, _,\
        _, _, _, _, _, _, _, _, \
        _, predict_test, _ ,\
        x_re, \
        pmu5_1, pvar5_1, pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de)
        return mu_latent, predict_test, x_re
