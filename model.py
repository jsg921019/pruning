import torch
import torch.nn as nn

default_cfg = [[64], [128], [256, 256], [512, 512], [512, 512]]

class CustomVGG(nn.Module):
  
    def __init__(self, cfg=None, bias=True):

        super(CustomVGG, self).__init__()

        if cfg is None:
            cfg = default_cfg
        
        self.features = self._build_features(cfg, bias)
        self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(cfg[-1][-1], 6)

    def _build_features(self, cfg, bias):

        layers = []
        in_channels = 3
        
        for stage in cfg:
            for out_channels in stage:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
class CustomVGGHalf(nn.Module):
  
    def __init__(self, cfg=None, bias=True):

        super(CustomVGGHalf, self).__init__()

        if cfg is None:
            cfg = default_cfg
        
        self.features = self._build_features(cfg, bias).half()
        self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(cfg[-1][-1], 6).half()

    def _build_features(self, cfg, bias):

        layers = []
        in_channels = 3
        
        for stage in cfg:
            for out_channels in stage:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.features(x.half())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x