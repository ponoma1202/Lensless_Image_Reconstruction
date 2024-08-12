import torch
import os
from tqdm import tqdm
import torchvision.transforms as transforms

from recon_transformer import Recon_Transformer
from convnext import ConvRecon
from dataset import get_loader
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 1

def main():

    dataset = "Mirflickr"
    # num_heads = 4
    # num_blocks = 6
    # embed_dim = 128            # dimension of embedding/hidden layer in Transformer
    # patch_size = 15     # 270 / 15 = 18
    # n_channels = 3
    # ffn_multiplier = 2
    # dropout_rate = 0.1
    min_side_len = 270
    num_workers = 4
    save_path = '/home/ponoma/workspace/Basic_Transformer/checkpoint_experiment/'
    infer_results = './infer_results_after_35_epochs/'      # NOTE: if in workspace, "." will not be in Basic_Transformer

    if not os.path.exists(infer_results):
        os.makedirs(infer_results)

    #model = Recon_Transformer(min_side_len, patch_size, n_channels, num_heads, num_blocks, embed_dim, ffn_multiplier, dropout_rate)
    model = ConvRecon()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
        model = model.to(device) 
    else:
        device = torch.device("cpu")
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth'), map_location=torch.device('cpu')))
    
    _, _, test_loader = get_loader(dataset, min_side_len, batch_size=1, num_workers=num_workers)

    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            input, target, img_name = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            output = transforms.ToPILImage()(output.squeeze())
            output.save(os.path.join(infer_results, img_name[0] + '.jpg'), format='JPEG')       # Test if this works

if __name__ == "__main__":
    main()