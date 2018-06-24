import torch
from torchvision.models import resnet34

class BilinearResNet34(torch.nn.Module):

  def __init__(self):
    """Declare all needed layers."""
    torch.nn.Module.__init__(self)
    resnet_model = resnet34(pretrained=True)

    self.conv1 = resnet_model.conv1
    self.bn1 = resnet_model.bn1
    self.relu = resnet_model.relu
    self.maxpool = resnet_model.maxpool
    self.layer1 = resnet_model.layer1
    self.layer2 = resnet_model.layer2
    self.layer3 = resnet_model.layer3
    self.layer4 = resnet_model.layer4

    # Linear classifier.
    self.fc = torch.nn.Linear(512**2, 12)

    # Initialize the fc layers.
    torch.nn.init.kaiming_normal_(self.fc.weight.data)
    if self.fc.bias is not None:
      torch.nn.init.constant_(self.fc.bias.data, val=0)

  def forward(self, X):
    N = X.size()[0]
    # assert X.size() == (N, 3, 448, 448)

    x = self.conv1(X)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x_size = x.size()
    feature_size = x_size[2]

    # assert X.size() == (N, 512, 14, 14)

    x = x.view(N, 512, feature_size**2)
    x = torch.bmm(x, torch.transpose(x, 1, 2)) / (feature_size**2)  # Bilinear

    # assert x.size() == (N, 512, 512)

    x = x.view(N, 512**2)
    x = torch.sqrt(x + 1e-5)
    x = torch.nn.functional.normalize(x)
    x = self.fc(x)

    # assert x.size() == (N, 200)

    return x