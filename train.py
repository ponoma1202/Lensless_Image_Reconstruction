import torch
from tqdm import tqdm
import os
import wandb

from convnext import ConvRecon
from recon_transformer import Recon_Transformer
from dataset import get_loader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 2

def main():
    debug = False
    convnext = True

    dataset = "Mirflickr"          
    batch_size = 8 
    num_epochs = 200
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
    save_path = './checkpoint_transformer/'
    if not debug:
        run = wandb.init(project='basic_transformer', config={"learning_rate":learning_rate,
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
        model =  ConvRecon() 
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

    train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs, train_psnr, train_ssim, val_psnr, val_ssim)
    if not debug:
        run.finish()


# Includes both training and validation
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs, train_psnr, train_ssim, val_psnr, val_ssim):
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
            print("New best model at epoch", epoch)
        torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch+1}.pth'))
        

def train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device, psnr, ssim): 
    model.train()
    total_mse = 0

    for step, batch in enumerate(tqdm(train_loader)):
        input, target, _ = batch
        input, target = input.to(device), target.to(device)   
        output = model(input)                                             
        optimizer.zero_grad()
        loss = criterion(output.squeeze(), target)                                   
        loss.backward()
        optimizer.step()        
        total_mse += loss.item()  
        with torch.no_grad():                                   # do not want to accumulate gradients for evaluation metrics
            psnr.update(input, target)        
            ssim.update(input, target)                               
        
    avg_mse = total_mse/ len(train_loader.dataset) 
    avg_psnr = psnr.compute()
    avg_ssim = ssim.compute() 
    psnr.reset()            # for next epoch
    ssim.reset() 
    print(f'Epoch {epoch+1}/{num_epochs}, Train MSE Loss: {avg_mse}, Train PSNR {avg_psnr}, Train SSIM {avg_ssim}')
    return avg_psnr, avg_mse, avg_ssim  


def validate(model, val_loader, criterion, device, save_path, psnr, ssim, load=False):
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
            psnr.update(input, target)
            ssim.update(input, target)                               
            total_loss += loss

        avg_mse = total_loss/len(val_loader.dataset)
        avg_psnr = psnr.compute()
        avg_ssim = ssim.compute()  
        psnr.reset()            # for next epoch
        ssim.reset()
        print(f'Val MSE Loss: {avg_mse}, Val PSNR: {avg_psnr}, Val SSIM: {avg_ssim} \n')
        return avg_psnr, avg_mse, avg_ssim
    

if __name__ == "__main__":
    main()
