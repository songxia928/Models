import torchvision
from torchvision import transforms 
import matplotlib.pyplot as plt


IMG_SIZE = 48  # = 28 时，报错。因为 下采样图片分辨率不是成倍减少

# PIL图像转tensor
pil_to_tensor=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),    # PIL图像尺寸统一  
    transforms.ToTensor()                       # PIL图像转tensor, (H,W,C) ->（C,H,W）,像素值[0,1]
])

# tensor转PIL图像
tensor_to_pil=transforms.Compose([
    transforms.Lambda(lambda t: t*255),  # 像素还原
    transforms.Lambda(lambda t: t.type(torch.uint8)),    # 像素值取整
    transforms.ToPILImage(),    # tensor转回PIL图像, (C,H,W) -> (H,W,C) 
])

# 数据集Pytorch Dataset
train_dataset=torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=pil_to_tensor) # 手写数字图像pillow(PIL),对应的数字



if __name__=='__main__':
    # 训练样本的tensor
    img_tensor,label=train_dataset[0]

    # 转回pil图像绘制
    plt.figure(figsize=(5,5))
    pil_img=tensor_to_pil(img_tensor)
    plt.imshow(pil_img)
    plt.show()


