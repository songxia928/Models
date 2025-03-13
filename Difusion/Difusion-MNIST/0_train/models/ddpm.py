import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
 



class DDPM(nn.Module):
    def __init__(self, model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.model = model.to(device)
 
        # register_buffer 可以提前保存alpha相关，节约时间
        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
 
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()
 
    def ddpm_schedules(self, beta1, beta2, T):
        '''
        提前计算各个step的alpha，这里beta是线性变化
        :param beta1: beta的下限
        :param beta2: beta的下限
        :param T: 总共的step数
        '''
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
 
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 # 生成beta1-beta2均匀分布的数组
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() # alpha累乘
 
        sqrtab = torch.sqrt(alphabar_t) # 根号alpha累乘
        oneover_sqrta = 1 / torch.sqrt(alpha_t) # 1 / 根号alpha
 
        sqrtmab = torch.sqrt(1 - alphabar_t) # 根号下（1-alpha累乘）
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
 
        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}} # 加噪标准差
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}  # 加噪均值
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }
 
    def forward(self, x, c):
        """
        训练过程中, 随机选择step和生成噪声
        """

        print(' -------- ddpm forward, x.shape, c.shape: ', x.shape, c.shape)
        print(' ---- x: ', x)
        print(' ---- c: ', c)

        # 随机选择step
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        print(' ---- _ts.shape, _ts: ', _ts.shape, _ts)
        # 随机生成正态分布噪声
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        # 加噪后的图像x_t
        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )
 
        # 将unet预测的对应step的正态分布噪声与真实噪声做对比
        return self.loss_mse(noise, self.model(x_t, c, _ts / self.n_T))
 
    def sample(self, n_sample, c, size, device):
        print(' -------- ddpm sample, c.shape: ', c.shape)
        print(' ---- n_sample: ', n_sample)
        print(' ---- c: ', c)


        # 随机生成初始噪声图片 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
 
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
 
            eps = self.model(x_i, c, t_is)
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i
 
 
