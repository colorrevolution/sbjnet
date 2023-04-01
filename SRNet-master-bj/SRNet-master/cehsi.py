import torch
from torchvision.models import vgg19
import cv2
class Vgg19(torch.nn.Module):
    def __init__(self):

        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        print(features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):

        results = []
        for ii, model in enumerate(self.features):
            x = model(x)

            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

imagepath = "media/pre-trained_result.png"
print(imagepath)
vgg = vgg19()


a = torch.ones(4,10,3,256)
print(torch.transpose(a,1,2).size())
b = torch.zeros(4,10,3,256)
torch.mean(torch.abs(a - b))
# c = torch.matmul(a,b)
print(torch.mean(torch.abs(a - b)))