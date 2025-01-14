import torch
import os
from tqdm import tqdm
import torchvision.transforms as transforms

from recon_transformer import Recon_Transformer
from convnext import ConvRecon
from swin_transformer import SwinRecon
from dataset import get_loader
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 1

def main():
    # options are: 'basic_transformer', 'convnext', 'swin'
    model_type = 'convnext' 

    dataset = "Mirflickr"
    num_heads_vit = 4
    num_heads_swin = [4, 8, 16, 32]
    num_blocks = 6
    embed_dim = 128            # dimension of embedding/hidden layer in Transformer
    patch_size = (15, 19)    
    n_channels = 3  #TODO change as needed
    ffn_multiplier = 2
    dropout_rate = 0.1
    num_workers = 4
    height = 210
    width = 380                                             
    save_path = '/home/ponoma/workspace/Basic_Transformer/checkpoint_convnext_rgb_fixed/best_model.pth'
    infer_results = '/home/ponoma/workspace/Basic_Transformer/experiment/'     

    if not os.path.exists(infer_results):
        os.makedirs(infer_results)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number) 
    else:
        device = torch.device("cpu")

    if model_type == 'convnext':
        model = ConvRecon(n_channels) 
    elif model_type == 'swin':
        model = SwinRecon(n_channels=n_channels, img_size=height*width, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads_swin)
    else:
        model = Recon_Transformer(height, width, patch_size, n_channels, num_heads_vit, num_blocks, embed_dim, ffn_multiplier, dropout_rate) 
    model.to(device)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number)
        model.load_state_dict(torch.load(save_path)) 
        model = model.to(device) 
    else:
        device = torch.device("cpu")
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    
    _, _, test_loader = get_loader(dataset, batch_size=1, num_workers=num_workers)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            input, target, img_name = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            # output = transforms.ToPILImage()(output.squeeze())
            # output.save(os.path.join(infer_results, img_name[0] + '.jpg'), format='JPEG')      

            if torch.isnan(output).any() or torch.isnan(target).any():
                print(f"Skipping batch {step} due to NaN values.")
                continue

            psnr_val = psnr(output, target) 
            ssim_val = ssim(output, target)   
            mse_val = torch.nn.functional.mse_loss(output, target, reduction="mean")  # Reduces by default to a scalar
            lpips_val = lpips(output, target)

            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_mse += mse_val.item()
            total_lpips += lpips_val.item()
            
            num_batches += 1

    mean_psnr = total_psnr / num_batches
    mean_ssim = total_ssim / num_batches
    mean_mse = total_mse / num_batches
    mean_lpips = total_lpips / num_batches

    # Print the results
    print("Average PSNR: ", mean_psnr)       
    print("Average SSIM: ", mean_ssim)
    print("Average MSE Loss: ", mean_mse)
    print("Average LPIPS: ", mean_lpips)

if __name__ == "__main__":
    main()