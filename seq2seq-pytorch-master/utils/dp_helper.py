import torch

def dpmin(choose, lamb=1, vmax=float("inf")):
	if not choose:
		return vmax
	elif len(choose) == 1:
		return choose[0]
	elif len(choose) == 2:
		a = choose[0]
		b = choose[1]
		x = ((1 - ((a - b) / lamb / 2).tanh()) / 2).detach()
		return a * x + b * (1-x) - lamb * (x * (x+1e-12).log() + (1-x) * (1-x+1e-12).log())
	else:
		raise NotImplementedError("len(choose) > 3 has not been implemented")

def dpmax_hard(choose, vmin=float("-inf")):
	if not choose:
		return vmin
	else:
		return torch.max(torch.stack(choose), dim=0)[0]
