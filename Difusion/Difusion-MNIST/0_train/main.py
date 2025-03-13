import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from models.unet import Unet
from models.ddpm import DDPM
 
 
class ImageGenerator(object):
    def __init__(self):
        '''
        初始化，定义超参数、数据集、网络结构等
        '''
        self.epoch = 1 # 20
        self.sample_num = 100
        self.batch_size = 256
        self.lr = 0.0001
        self.n_T = 400
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_dataloader()
        self.sampler = DDPM(model=Unet(in_channels=1), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.sampler.model.parameters(), lr=self.lr)
        self.writer = SummaryWriter(log_dir='./output/summary')

        self.dir_out = './output'
        self.dir_models_saved = self.dir_out + '/models'
        self.dir_imgs_saved = self.dir_out + '/imgs'

        if not os.path.exists(self.dir_models_saved):
            os.makedirs(self.dir_models_saved)
        if not os.path.exists(self.dir_imgs_saved):
            os.makedirs(self.dir_imgs_saved)

    def init_dataloader(self):
        '''
        初始化数据集和dataloader
        '''
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = MNIST('./data/',
                              train=True,
                              download=True,
                              transform=tf)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataset = MNIST('./data/',
                            train=False,
                            download=True,
                            transform=tf)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
 
    def train(self):
        self.sampler.train()   # ddpm
        print('训练开始!!')
        n_iter = 0 
        for epoch in range(self.epoch):
            self.sampler.model.train()   # unet
            loss_mean = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = F.one_hot(labels, num_classes=10).float()

                # ==== forward
                # 将latent和condition拼接后输入网络
                loss = self.sampler(images, labels)

                loss_mean += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/train', loss, n_iter) 
                self.writer.add_scalar('Loss_mean/train', loss_mean, n_iter) 
                n_iter += 1

            train_loss = loss_mean / len(self.train_dataloader)
            print('epoch:{}, loss:{:.4f}'.format(epoch, train_loss))
            self.visualize_results(epoch)

        torch.save(self.sampler, self.dir_models_saved + '/sd.pt') 


    @torch.no_grad()
    def visualize_results(self, epoch):
        self.sampler.eval()
        
        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        labels = F.one_hot(torch.Tensor(np.repeat(np.arange(10), 10)).to(torch.int64), num_classes=10).to(self.device).float()

        # ==== sample
        out = self.sampler.sample(tot_num_samples, labels, (1, 28, 28), self.device)

        save_image(out, os.path.join(self.dir_imgs_saved, '{}.jpg'.format(epoch)), nrow=image_frame_dim)
 
 
 
if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()


