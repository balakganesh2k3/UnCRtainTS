"""
Synthetic smoke test + step-time / GPU-memory benchmark (task S2-2). No dataset needed.
Run from UnCRtainTS/model, with the same model flags you will train with, e.g.

  python time_full.py --covmode full --loss MGNLL --use_sar --batch_size 4 --steps 20
  python time_full.py --covmode diag --loss MGNLL --use_sar --batch_size 4 --steps 20   # reference

Prints seconds/step, peak GPU memory, loss and gradient norm.
"""
import os, sys, time
import torch

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
from parse_args import create_parser
from src import utils
from src.model_utils import get_model, init_full_cov_head
from src.learning.weight_init import weight_init

S2_BANDS, S1_BANDS = 13, 2

parser = create_parser(mode='train')
parser.add_argument('--steps', type=int, default=20, help='timed optimisation steps')
parser.add_argument('--warmup', type=int, default=3, help='untimed warm-up steps')
parser.add_argument('--patch', type=int, default=256, help='spatial size of the synthetic patches')
parser.add_argument('--mean_only_steps', type=int, default=0, help='first N steps use the MSE-only warm-up (full mode)')
config = utils.str2list(parser.parse_args(), list_args=["encoder_widths", "decoder_widths", "out_conv"])

# same channel bookkeeping as train_reconstruct.py
if config.loss in ['GNLL', 'MGNLL']:
    if config.loss == 'GNLL': config.covmode = 'uni'
    if config.covmode == 'iso':               config.out_conv[-1] += 1
    elif config.covmode in ['uni', 'diag']:   config.out_conv[-1] += S2_BANDS; config.var_nonLinearity = 'softplus'
    elif config.covmode == 'full':            config.out_conv[-1] += S2_BANDS * (S2_BANDS + 1) // 2; config.var_nonLinearity = 'softplus'
if not torch.cuda.is_available(): config.device = 'cpu'
device = torch.device(config.device)
print(f'device: {device}, covmode: {config.covmode}, batch_size: {config.batch_size}, patch: {config.patch}')

model = get_model(config).to(device)
model.netG.apply(weight_init)
if config.covmode == 'full': init_full_cov_head(model, config)
model.netG.train()
n_params = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)
print(f'trainable parameters: {n_params:,}')

B, T, P = config.batch_size, config.input_t, config.patch
c_in = S2_BANDS + S1_BANDS * int(config.use_sar)

def make_batch():
    A = torch.rand(B, T, c_in, P, P)
    # small, learnable synthetic target (residuals of a realistic size, ~0.05), instead of pure noise
    target = 0.1 * A[:, :, -S2_BANDS:].mean(dim=1, keepdim=True)
    return {'A': A,
            'B': target,
            'dates': (torch.arange(T).float() * 5 + torch.rand(B, T) * 2),
            'masks': torch.zeros(B, T, P, P)}

if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
times = []
for step in range(config.warmup + config.steps):
    model.mean_only = (config.covmode == 'full' and step < config.mean_only_steps)
    model.set_input(make_batch())
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    model.optimize_parameters()
    if device.type == 'cuda': torch.cuda.synchronize()
    dt = time.time() - t0
    loss = model.loss_G.item()
    if not (loss == loss and abs(loss) != float('inf')):
        print(f'step {step:3d}  loss is NOT finite ({loss}); stopping. Try a smaller --lr (e.g. 1e-4), --grad_clip 1.0, or --mean_only_steps.')
        break
    if step >= config.warmup: times.append(dt)
    print(f'step {step:3d}  loss {loss:10.4f}  grad_norm {float(model.grad_norm):9.3f}  {dt:6.2f}s')

var = model.netG.variance
print('\ncovariance output shape:', None if var is None else tuple(var.shape), '(expected [B,1,13,13,H,W] for full)')
if not times:
    print('\nNo timed steps completed (diverged during warm-up); no timing available.'); sys.exit(1)
mean_t = sum(times) / len(times)
print(f'\nseconds/step: {mean_t:.3f}   seconds/sample: {mean_t / B:.3f}')
if device.type == 'cuda': print(f'peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
print('To estimate an epoch: (number of training samples / batch_size) * seconds/step')