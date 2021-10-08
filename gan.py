# Modified by CartoonGAN and DCGAN
import os, torch, time, cv2
from PIL import Image
from torch.autograd.variable import Variable
from torch.serialization import load
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import numpy as np

Setting = {
    "EPOCHS": 50,#150
    "BATCH_SIZE": 8,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "DEVICE": torch.device("cpu"),
    "LATENT_SIZE": 100,
    "IMAGE_SIZE": 256,
    "LR": 0.0002,
    "CHANNELS": 3,
    "beta1": 0.5
}

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def data_load(
    path, subfolder = None,
    transform=transforms.Compose([
        transforms.Resize(Setting["IMAGE_SIZE"]),
        transforms.CenterCrop(Setting["IMAGE_SIZE"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), batch_size=Setting["BATCH_SIZE"], shuffle=False, drop_last=False
):
    dset = datasets.ImageFolder(root=path, transform=transform)

    if subfolder is not None:
        ind = dset.class_to_idx[subfolder]
        n = 0
        for i in range(dset.__len__()):
            if ind != dset.imgs[n][1]:
                del dset.imgs[n]
                n -= 1

            n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class Generator(nn.Module):
    def __init__(self, in_nc = 3, out_nc = 3, nf=64, nb=6):
        super(Generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3), #k7n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), #k3n128s2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), #k3n256s1
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), #k3n256s1
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), #k3n128s1/2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), #k3n64s1/2
            nn.Conv2d(nf, nf, 3, 1, 1), #k3n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh(),
        )

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        output = self.up_convs(x)

        return output


class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc = 3, out_nc = 1, nf=32):
        super(Discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


G = Generator().to(Setting["DEVICE"])
D = Discriminator().to(Setting["DEVICE"])
G.apply(initialize_weights)
D.apply(initialize_weights)

def output(path, name="finish.png"):
    input_image = Image.open(path).convert("RGB")
    input_image = np.asarray(input_image)
    input_image2 = input_image[:,:,[2,1,0]]

    # input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = transforms.ToTensor()(input_image)
    input_image = -1 + 2 * input_image
    
    # input_image2 = transforms.ToTensor()(input_image2).unsqueeze(0)
    input_image2 = transforms.ToTensor()(input_image2)
    input_image2 = -1 + 2 * input_image2
    
    input_image = torch.stack([input_image, input_image2], dim=0)

    with torch.no_grad():
        if str(Setting["DEVICE"]) == "cpu":
            input_image = Variable(input_image).float()
        else:
            input_image = Variable(input_image).cuda()

        output_image = G(input_image)
        save_image(output_image * 0.5 + 0.5, os.path.join("./output", name))

def training():
    loader = data_load("./data", "train")
    
    BCE_loss = nn.BCELoss().to(Setting["DEVICE"])

    D_optimizer = optim.Adam(D.parameters(), lr=Setting["LR"], betas=(Setting["beta1"], 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=Setting["LR"], betas=(Setting["beta1"], 0.999))
    
    real_origin = torch.ones(Setting["BATCH_SIZE"], 1, Setting["IMAGE_SIZE"] // 4, Setting["IMAGE_SIZE"] // 4).to(Setting["DEVICE"]).to(Setting["DEVICE"])
    fake_origin = torch.zeros(Setting["BATCH_SIZE"], 1, Setting["IMAGE_SIZE"] // 4, Setting["IMAGE_SIZE"] // 4).to(Setting["DEVICE"]).to(Setting["DEVICE"])

    print("Training start.")
    start_training = time.time()

    for epoch in range(Setting["EPOCHS"]):
        start_epoch = time.time()

        for i, (y, _) in enumerate(loader):
            D.zero_grad()
            real, fake = real_origin, fake_origin
            start_loop = time.time()
            
            if y.shape[0] != real.shape[0]:
                real = torch.ones(y.shape[0], 1, Setting["IMAGE_SIZE"] // 4, Setting["IMAGE_SIZE"] // 4).to(Setting["DEVICE"]).to(Setting["DEVICE"])
                fake = torch.zeros(y.shape[0], 1, Setting["IMAGE_SIZE"] // 4, Setting["IMAGE_SIZE"] // 4).to(Setting["DEVICE"]).to(Setting["DEVICE"])
                if i == 0:
                  real_origin, fake_origin = real, fake
            
            # x = torch.randn(y.shape[0], Setting["CHANNELS"], Setting["IMAGE_SIZE"], Setting["IMAGE_SIZE"])
            # x, y = x.to(Setting["DEVICE"]), y.to(Setting["DEVICE"])
            y = y.to(Setting["DEVICE"])

            D_real = D(y)
            D_real_loss = BCE_loss(D_real, real)
            D_real_loss.backward()
            D_x = D_real.mean().item()

            # G_ = G(x)
            G_ = G(y)
            D_fake = D(G_.detach())
            D_fake_loss = BCE_loss(D_fake, fake)
            D_fake_loss.backward()
            D_G_fake = D_fake.mean().item()
            
            Disc_loss = D_real_loss + D_fake_loss
            D_optimizer.step()

            G.zero_grad()
            D_fake = D(G_)
            G_loss = BCE_loss(D_fake, real)
            G_loss.backward()
            D_G_fake2 = D_fake.mean().item()
            G_optimizer.step()

            if i % 50 == 0:
                print("[{}/{}] [{}/{}] - time: {:.2f}, Loss: {:.3f}, D(x): {:.5f}, D(G(x)): {:.5f}/{:.5f}".format(epoch, Setting["EPOCHS"], i, len(loader), time.time() - start_loop, Disc_loss.item(), G_loss.item(), D_x, D_G_fake, D_G_fake2))
            # print("[{}/{}] [{}/{}] - time: {:.2f}, Disc Loss: {:.3f}, G Loss: {:.3f}, D(x): {:.5f}, D(G(x)): {:.5f}".format(epoch + 1, Setting["EPOCHS"], i + 1, len(loader), time.time() - start_loop, Disc_loss.item(), G_loss.item(), D_real_loss.item(), D_fake_loss.item()))
          
        print("[{}/{}] total time: {:.2f}".format(epoch + 1, Setting["EPOCHS"], i + 1, len(loader), time.time() - start_epoch))
        save()

    save()
    print("Training finished - time: {:2f}".format(time.time() - start_training))

def save():
    torch.save(G.state_dict(), "./model/Generator.pt")
    torch.save(D.state_dict(), "./model/Discriminator.pt")

def load():
    G.load_state_dict(torch.load("./model/Generator.pt"))
    D.load_state_dict(torch.load("./model/Discriminator.pt"))
    print("loaded")

def video(name="result.avi"):
    video = cv2.VideoCapture("hamilton_clip.mp4")
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    size = (int(width), int(height))

    out = cv2.VideoWriter(os.path.join("./output", name), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    while video.isOpened():
        ret, frame = video.read()
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = frame[:,:,[2,1,0]]
        frame = transforms.ToTensor()(frame).unsqueeze(0)
        frame = -1 + 2 * frame
        
        with torch.no_grad():
            if str(Setting["DEVICE"]) == "cpu":
                frame = Variable(frame).float()
            else:
                frame = Variable(frame).cuda()
        
            frame = G(frame) * 0.5 + 0.5
            image = transforms.ToPILImage()(frame[0])
            frame = np.asarray(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, size)
            out.write(frame)

    video.release()
    out.release()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # video.release()

if __name__ == "__main__":
    print("Using Device:", Setting["DEVICE"])
    if os.path.exists("./model/Generator.pt"): load()
    # training()
    # output()

    target = "./input"
    for f in os.listdir(target):
        if os.path.isfile(os.path.join(target, f)):
            output(os.path.join(target, f), f)
    
    video()
