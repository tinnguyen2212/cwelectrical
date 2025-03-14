# Đặt nội dung này vào file __init__.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths

# Đăng ký đường dẫn cho alpha models
models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)
folder_paths.folder_names_and_paths["flux_alpha_models"] = ([models_dir], folder_paths.supported_pt_extensions)

# ===== ĐỊNH NGHĨA KIẾN TRÚC ALPHANET =====
# Sao chép từ script huấn luyện

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool + double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling + double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Chỉnh size để khớp với x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AlphaUNet(nn.Module):
    def __init__(self, rgb_channels=3, latent_channels=4, bilinear=True):
        super(AlphaUNet, self).__init__()
        self.rgb_channels = rgb_channels
        self.latent_channels = latent_channels
        self.bilinear = bilinear
        
        # Số filters ở mỗi tầng
        base_filters = 64
        
        # RGB branch
        self.rgb_inc = DoubleConv(rgb_channels, base_filters)
        self.rgb_down1 = Down(base_filters, base_filters * 2)
        self.rgb_down2 = Down(base_filters * 2, base_filters * 4)
        self.rgb_down3 = Down(base_filters * 4, base_filters * 8)
        
        # Latent branch
        self.latent_conv = nn.Conv2d(latent_channels, base_filters * 2, kernel_size=1)
        
        # Middle fusion
        factor = 2 if bilinear else 1
        self.middle = DoubleConv(base_filters * 8 + base_filters * 2, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output - 1 kênh alpha
        self.outc = nn.Sequential(
            nn.Conv2d(base_filters, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb, latent):
        # RGB branch
        x1 = self.rgb_inc(rgb)
        x2 = self.rgb_down1(x1)
        x3 = self.rgb_down2(x2)
        x4 = self.rgb_down3(x3)
        
        # Xử lý latent
        lat = self.latent_conv(latent)
        
        # Đảm bảo kích thước latent khớp với x4
        if lat.shape[2:] != x4.shape[2:]:
            lat = F.interpolate(
                lat, 
                size=x4.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Kết hợp features
        x = torch.cat([x4, lat], dim=1)
        x = self.middle(x)
        
        # Decoder
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output alpha channel
        alpha = self.outc(x)
        
        return alpha
# Node để tải model
class FluxAlphaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("flux_alpha_models"), )}}
    
    RETURN_TYPES = ("FLUX_ALPHA_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "FLUX"
    
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("flux_alpha_models", model_name)
        predictor = FluxAlphaPredictor(model_path)
        return (predictor,)

# Node để tạo alpha từ RGB và latent
class FluxRGBACreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha_model": ("FLUX_ALPHA_MODEL",),
            },
            "optional": {
                "samples": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_rgba"
    CATEGORY = "FLUX"
    
    def create_rgba(self, images, alpha_model, samples=None):
        # Chuyển model lên GPU
        device = images.device
        alpha_model.to_device(device)
        
        # Lấy latent nếu có
        latents = samples["samples"] if samples is not None else None
        
        # Dự đoán alpha channel
        alpha = alpha_model.predict_alpha(images, latents)
        
        # Kết hợp RGB và alpha
        rgba = torch.cat([images, alpha], dim=3)
        
        return (rgba,)

# Node để lưu ảnh RGBA
class SaveFluxRGBA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "flux_rgba"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_rgba"
    OUTPUT_NODE = True
    CATEGORY = "image"
    
    def save_rgba(self, images, filename_prefix):
        # Kiểm tra số kênh
        if images.shape[3] != 4:
            raise ValueError(f"Expected RGBA image with 4 channels, got {images.shape[3]}")
        
        # Thư mục output
        output_dir = folder_paths.get_output_directory()
        rgba_dir = os.path.join(output_dir, "rgba")
        os.makedirs(rgba_dir, exist_ok=True)
        
        results = []
        for i, img in enumerate(images):
            # Convert to numpy
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            
            # Tạo ảnh PIL và lưu
            img_pil = Image.fromarray(img_np, mode='RGBA')
            filename = f"{filename_prefix}_{i:05d}.png"
            filepath = os.path.join(rgba_dir, filename)
            img_pil.save(filepath)
            
            results.append({
                "filename": filename,
                "subfolder": "rgba",
                "type": "output"
            })
        
        return {"ui": {"images": results}}

# Đăng ký các node
NODE_CLASS_MAPPINGS = {
    "FluxAlphaModelLoader": FluxAlphaModelLoader,
    "FluxRGBACreator": FluxRGBACreator,
    "SaveFluxRGBA": SaveFluxRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAlphaModelLoader": "Load FLUX Alpha Model",
    "FluxRGBACreator": "Create FLUX RGBA",
    "SaveFluxRGBA": "Save FLUX RGBA",
}