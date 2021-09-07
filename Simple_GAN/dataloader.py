import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def Dataload(batch_size=64, transforms=transforms):
    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader