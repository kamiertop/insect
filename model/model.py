import torch.nn as nn

from torchvision import models


class Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = get_insect_model(num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_insect_model(num_classes: int, fine_tune: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if not fine_tune:
        # 冻结预训练模型参数，仅训练最后的分类头
        for param in model.parameters():
            param.requires_grad = False
    else:
        # 先冻结全部层，再只解冻后两层做微调，避免参数状态不一致
        for param in model.parameters():
            param.requires_grad = False
        for name, child in model.named_children():
            if name in ["layer3", "layer4"]:
                for param in child.parameters():
                    param.requires_grad = True

    # torchvision 的 fc 在 ResNet50 上应为 Linear，这里显式校验保证类型安全
    if not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected model.fc to be nn.Linear for ResNet50")

    in_features: int = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )

    return model
