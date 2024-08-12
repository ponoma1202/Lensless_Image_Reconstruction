import torch
from tqdm import tqdm
import os
import wandb

from classification_model import Transformer
from convnext import ConvRecon
from recon_transformer import Recon_Transformer
from dataset import get_loader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 0

def main():
    debug = True

    dataset = "Mirflickr"          
    batch_size = 8 
    num_epochs = 200
    learning_rate = 5e-4
    num_classes = 10
    num_heads = 4
    num_blocks = 6
    embed_dim = 128            # dimension of embedding/hidden layer in Transformer
    patch_size = 15     # 270 / 15 = 18
    n_channels = 3
    warmup_epochs = 10
    ffn_multiplier = 2
    min_side_len = 270
    dropout_rate = 0.1
    num_workers = 4
    save_path = './checkpoint_experiment/'
    if not debug:
        run = wandb.init(project='basic_transformer', config={"learning_rate":learning_rate,
                                                        "architecture": Transformer,
                                                        "dataset": dataset,
                                                        "epochs":num_epochs,
                                                        "batch_size":batch_size,
                                                        "classes":num_classes,
                                                        "num_heads":num_heads,
                                                        "num_encoder_layers": num_blocks,
                                                        "embed_dim":embed_dim,
                                                        "patch_size": patch_size,
                                                        "n_channels": n_channels,
                                                        "warmup_epochs":warmup_epochs,
                                                        "ffn_multiplier": ffn_multiplier, 
                                                        "dropout_rate": dropout_rate})     

    # Get data loaders
    train_loader, val_loader, _ = get_loader(dataset, min_side_len, batch_size, num_workers)

    # See if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number) 
    else:
        device = torch.device("cpu")
    
    # Initialize model and move to GPU if available
    #model = Recon_Transformer(min_side_len, patch_size, n_channels, num_heads, num_blocks, embed_dim, ffn_multiplier, dropout_rate)
    model =  ConvRecon()    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)         
    criterion = torch.nn.MSELoss()          # for reconstruction

    if not os.path.exists(save_path):          
        os.mkdir(save_path)

    train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs)
    if not debug:
        run.finish()


# Includes both training and validation
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, debug, warmup_epochs):
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_epochs, end_factor=1.0, total_iters=warmup_epochs-1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs-warmup_epochs, eta_min=1e-5)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Start training epoch {epoch+1}/{num_epochs}...')
        train_accuracy, train_loss = train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device) 
        val_acc, val_loss = validate(model, val_loader, criterion, device, save_path, debug)
        if not debug:
            wandb.log({"training_loss":train_loss, "validation_loss":val_loss, "epoch":epoch, "learning rate":optimizer.param_groups[-1]['lr']})
        if epoch < warmup_epochs:
            linear_warmup.step()
        else:
            scheduler.step()

        # save best model
        if val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
        

def train_epoch(model, epoch, num_epochs, train_loader, optimizer, criterion, device):
    model.train()
    total_correct = 0
    total_loss = 0

    for step, batch in enumerate(tqdm(train_loader)):
        input, target, _ = batch
        input, target = input.to(device), target.to(device)   
        output = model(input)                                             
        optimizer.zero_grad()
        loss = criterion(output.squeeze(), target)                                      
        loss.backward()
        optimizer.step()        
        total_loss += loss.item()
        total_correct += (output == target).sum().item()                                  # summing over a list results in a list so need to use .item() to get a number.

    accuracy = total_correct / len(train_loader.dataset)
    avg_loss = total_loss/ len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {accuracy}')
    return accuracy, avg_loss          


def validate(model, val_loader, criterion, device, save_path, debug, load=False):
    if load:
        model.load_state_dict(torch.load(save_path))                         # if loading from saved model
    
    model.eval()
    print("Starting validation...")
    with torch.no_grad():
        total_correct = 0
        total_loss = 0.0

        for step, batch in enumerate(tqdm(val_loader)):
            input, target, _ = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output.squeeze(), target)                                 
            total_loss += loss
            total_correct += (output == target).sum().item()         
        accuracy = total_correct/len(val_loader.dataset)
        avg_loss = total_loss/len(val_loader.dataset)
        print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy} \n')
        return accuracy, avg_loss
    

if __name__ == "__main__":
    main()
