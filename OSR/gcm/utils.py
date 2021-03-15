import numpy as np
import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F

def sample_gaussian(m, v):
	sample = torch.randn(m.shape).cuda()
	z = m + (v**0.5)*sample
	return z

def gaussian_parameters(h, dim=-1):
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl


def get_exp_name(opt):
	cf_option_str = [opt.dataset]

	# if is the baseline
	if opt.baseline:
		additional_str = 'baseline'
		cf_option_str.append(additional_str)

	if opt.lamda:
		additional_str = 'lamda%s' %opt.lamda
		cf_option_str.append(additional_str)

	if opt.lr and opt.lr != 0.001:
		additional_str = 'lr%s' %opt.lr
		cf_option_str.append(additional_str)

	if opt.beta_z != 1:
		additional_str = 'betaz%s' %opt.beta_z
		cf_option_str.append(additional_str)


	if opt.encode_z:
		additional_str = 'encodez%s' %opt.encode_z
		cf_option_str.append(additional_str)

	if opt.contrastive_loss:
		additional_str = 'contra'
		cf_option_str.append(additional_str)

	if opt.contrastive_loss and opt.temperature:
		additional_str = 'T%s' %opt.temperature
		cf_option_str.append(additional_str)

	# if opt.lr_decay:
	# 	additional_str = 'lrdecay%s' %opt.lr_decay
	# 	cf_option_str.append(additional_str)

	if opt.wd != 0.00:
		additional_str = 'wd%s' %opt.wd
		cf_option_str.append(additional_str)


	if opt.debug:
		additional_str = 'debug'
		cf_option_str.append(additional_str)

	return "-".join(cf_option_str)


def mixup_data(x, y, alpha, device):
	"""Returns mixed inputs, pairs of targets, and lambda"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).to(device)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		# pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))