import torch

from src.models.backbones import (
    get_FlowNetS_model,
    get_FlowNetSD_model,
    get_FlowNetC_model,
    get_GeoFlowNet_model,
    get_GeoFlowNetC_model,
    get_GeoFlowNetCNoCorr_model,
    get_GeoFlowNetNoUp_model,
)

MODEL_MAPPING = {
    "flownets": get_FlowNetS_model,
    "flownetsd": get_FlowNetSD_model,
    "flownetc": get_FlowNetC_model,
    "flownetcnocorr": get_GeoFlowNetCNoCorr_model,
    "geoflownet": get_GeoFlowNet_model,
    "geoflownetc": get_GeoFlowNetC_model,
    "geoflownetcnocorr": get_GeoFlowNetCNoCorr_model,
    "geoflownetnoup": get_GeoFlowNetNoUp_model,
}

def get_backbone(backbone_name, batch_norm):
    """
    Get the specified backbone model.
    """
    backbone_name = backbone_name.lower()
    model_fn = MODEL_MAPPING.get(backbone_name)
    
    if model_fn is None:
        raise ValueError(f"Unknown backbone name: {backbone_name}")
        
    return model_fn(batch_norm)

def load_backbone(backbone_name, use_batch_norm, device, pretrained_weights_path = None):
    """
    Load a backbone model and optionally initialize it with pre-trained weights.
    """
    model = get_backbone(backbone_name, use_batch_norm).to(device)
    
    if pretrained_weights_path:
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device)["model_state_dict"])
        
    print(f"Model name: {backbone_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model



