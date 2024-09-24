import torch
from tqdm import tqdm
import os
import wandb

from convnext import ConvRecon
from recon_transformer import Recon_Transformer
from dataset import get_loader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 2

def main():
    debug = False
    convnext = True
    save_every_epoch = True

    dataset = "Mirflickr"          
    batch_size = 8 
    learning_rate = 5e-4
    num_heads = 4
    num_blocks = 6
    embed_dim = 128            # dimension of embedding/hidden layer in Transformer
    patch_size = (15, 19)     # 210 / 15 = 14 and 380 / 19 = 20
    n_channels = 3
    warmup_epochs = 10
    ffn_multiplier = 2
    height = 210
    width = 380
    dropout_rate = 0.1
    num_workers = 4
    save_path = './checkpoint_convnext_rgb_fixed/'

    if convnext:
        num_epochs = 35
    else:
        num_epochs = 200
    
    if not debug:
        run = wandb.init(project='convnext', config={"learning_rate":learning_rate,
                                                        "architecture": Recon_Transformer,
                                                        # "dataset": dataset,
                                                        "epochs":num_epochs,
                                                        "batch_size":batch_size,
                                                        # "num_heads":num_heads,
                                                        # "num_encoder_layers": num_blocks,
                                                        # "embed_dim":embed_dim,
                                                        # "patch_size": patch_size,
                                                        "n_channels": n_channels,
                                                        "warmup_epochs":warmup_epochs, })
                                                        # "ffn_multiplier": ffn_multiplier, 
                                                        # "dropout_rate": dropout_rate})     

    # Get data loaders
    train_loader, val_loader, _ = get_loader(dataset, batch_size, num_workers)

    # See if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number) 
    else:
        device = torch.device("cpu")
    
    # Initialize model and move to GPU if available

    if convnext:
        model = ConvRecon(n_channels) 
    else:
        model = Recon_Transformer(height, width, patch_size, n_channels, num_heads, num_blocks, embed_dim, ffn_multiplier, dropout_rate) 
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)         
    criterion = torch.nn.MSELoss()          # for reconstruction

    # Training metrics
    train_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Validation metrics
    val_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if not os.path.exists(save_path):          
        os.mkdir(save_path)

    train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs, train_psnr, train_ssim, val_psnr, val_ssim, save_every_epoch)
    if not debug:
        run.finish()


# Includes both training and validation
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs, train_psnr, train_ssim, val_psnr, val_ssim, save_every_epoch):
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_epochs, end_factor=1.0, total_iters=warmup_epochs-1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs-warmup_epochs, eta_min=1e-5) 

    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Start training epoch {epoch+1}/{num_epochs}...')
        train_psnr_out, train_mse_loss, train_ssim_out = train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device, train_psnr, train_ssim) 
        val_psnr_out, val_mse_loss, val_ssim_out = validate(model, val_loader, criterion, device, save_path, val_psnr, val_ssim)
        if not debug:
            wandb.log({"training_mse_loss":train_mse_loss, 
                       "training_psnr": train_psnr_out,
                       "train_ssim": train_ssim_out,
                       "validation_loss": val_mse_loss,
                       "val_psnr": val_psnr_out, 
                       "val_ssim": val_ssim_out,
                       "epoch":epoch, 
                       "learning rate":optimizer.param_groups[-1]['lr']})
        if epoch < warmup_epochs:
            linear_warmup.step()
        else:
            scheduler.step()

        # save best model
        if val_mse_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print("New best model at epoch", epoch, "\n")
        if save_every_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch+1}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
        

def train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device, train_psnr, train_ssim): 
    model.train()
    total_mse = 0
    my_psnr = 0.0
    my_ssim = 0.0

    for step, batch in enumerate(tqdm(train_loader)):
        input, target, _ = batch
        input, target = input.to(device), target.to(device)   
        output = model(input)                                             
        optimizer.zero_grad()
        loss = criterion(output.squeeze(), target.squeeze())                                   
        loss.backward()
        optimizer.step()        
        total_mse += loss.item() 

        # output = output.detach().cpu().numpy()  # TODO remove these lines
        # target = target.detach().cpu().numpy()        
        with torch.no_grad():                                   # do not want to accumulate gradients for evaluation metrics
            # for i in range(input.shape[0]):
            #     curr_psnr = psnr(target[i], output[i], data_range=1.0) #10 * np.log10((max_pixel_val ** 2) / total_mse)
            #     curr_ssim = ssim(target[i], output[i], data_range=1.0, channel_axis=0)
            #     my_ssim += curr_ssim 
            #     my_psnr += curr_psnr
            train_psnr.update(output, target)        
            train_ssim.update(output, target)  
        total_mse *= input.shape[0]                             # multiply avg batch mse by batch size to get total mse for the batch                    
        
    avg_mse = total_mse / len(train_loader.dataset) 
    # avg_ssim = my_ssim / len(train_loader.dataset)
    #avg_psnr = my_psnr / len(train_loader.dataset)
    avg_psnr = train_psnr.compute()
    avg_ssim = train_ssim.compute() 
    train_psnr.reset()            # for next epoch
    train_ssim.reset() 
    print(f'Epoch {epoch+1}/{num_epochs}, Train MSE Loss: {avg_mse}, Train PSNR {avg_psnr}, Train SSIM {avg_ssim}')
    return avg_psnr, avg_mse, avg_ssim  


def validate(model, val_loader, criterion, device, save_path, val_psnr, val_ssim, load=False):
    if load:
        model.load_state_dict(torch.load(save_path))                  
    
    model.eval()
    print("Starting validation...")
    with torch.no_grad():
        total_loss = 0.0

        for step, batch in enumerate(tqdm(val_loader)):
            input, target, _ = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output.squeeze(), target)  
            total_loss += loss 
            val_psnr.update(output, target)
            val_ssim.update(output, target)                               

        avg_mse = total_loss/len(val_loader.dataset)
        avg_psnr = val_psnr.compute()
        avg_ssim = val_ssim.compute()  
        val_psnr.reset()            # for next epoch
        val_ssim.reset()
        print(f'Val MSE Loss: {avg_mse}, Val PSNR: {avg_psnr}, Val SSIM: {avg_ssim} \n')
        return avg_psnr, avg_mse, avg_ssim
    

if __name__ == "__main__":
    main()
