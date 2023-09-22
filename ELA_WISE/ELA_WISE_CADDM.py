import torch


class EnhancedCADDM(nn.Module):
    def __init__(self):
        super(EnhancedCADDM, self).__init__()
        
        self.caddm_model = CADDM(num_classes=2, backbone='efficientnet-b4')
        self.my_feature_model = EfficientNet.from_pretrained('efficientnet-b4')
        self.fc = nn.Linear(self.caddm_model.inplanes * 2, 2)  # 兩次特徵的連接
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, feature_caddm,feature_yours):
        _, caddm_feat = self.caddm_model.base_model(feature_caddm)
        _, my_feature = self.my_feature_model(feature_yours)
        combined_feat = torch.cat([caddm_feat, my_feature], dim=1)
        out = self.fc(combined_feat)
        predict=self.softmax(out)
        return predict
