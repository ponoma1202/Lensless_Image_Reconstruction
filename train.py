import torch
from model import Transformer
import torchvision
from utils import Rescale
from tqdm import tqdm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
gpu_number = 1

def main():
    # Use CIFAR 10 for the data
    batch_size = 32
    num_epochs = 10
    save_path = './checkpoint/model.pth'

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),                  # issue: scales image to [0,1] range          
                                                # do not want to center around 0. center around mean instead to avoid negative values (for embedding)
                                                torchvision.transforms.Normalize((0, 0, 0), (0.247, 0.243, 0.261)),       # from https://github.com/kuangliu/pytorch-cifar/issues/19  
                                                torchvision.transforms.Grayscale(), 
                                                Rescale()])      # resolve scaling issue from ToTensor (embedding only takes in ints, so we need [0, 255] range)      
                                          
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    in_dim = 256                # Number of embeddings in input sequence (size of input vocabulary). Pixels have range [0, 255]
    out_dim = 10                # Number of embeddings in output sequence. Each image is labeled with a single class (10 total classes)
    
    # See if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number) 
    else:
        device = torch.device("cpu")
    
    # Initialize model and move to GPU if available
    model = Transformer(in_dim, out_dim, device)
    model.to(device)

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)           # TODO: potentially add learning rate scheduler later and change parameters to paper's params
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(save_path):          
        os.mkdir(save_path)

    train(model, trainloader, optimizer, criterion, num_epochs, device, save_path)
    validate(model, testloader, criterion, device, save_path)


def train(model, trainloader, optimizer, criterion, num_epochs, device, save_path):
    model.train()
    
    for epoch in range(num_epochs):
        total_correct = 0
        total = 0
        for step, batch in enumerate(tqdm(trainloader)):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input, target)                                                    # result is a (num_classes, batch_size) tensor
            optimizer.zero_grad()
            loss = criterion(output.squeeze(), target)                                       # take argmax to get the class with the highest "probability"
            loss.backward()
            optimizer.step() 
            pred = output.squeeze().argmax(dim=1)                                       # output is (batch_size, target seq len, num_classes) so need to squeeze to (batch_size, num_classes). For classification, target seq len = 1
            total += target.size(0)                             # get batch size
            total_correct += (pred == target).sum().item()                              # summing over a list results in a list so need to use .item() to get a number.
        print("Total number correct: ", total_correct)
        print("Total number of images", len(trainloader.dataset))
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {total_correct / len(trainloader.dataset)}')
    
    torch.save(model.state_dict(), save_path)         

def validate(model, testloader, criterion, device, save_path, load=False):
    if load:
        model.load_state_dict(torch.load(save_path))                         # if loading from saved model
    
    model.eval()
    print("Starting validation...")
    with torch.no_grad():
        total_correct = 0
        for step, batch in enumerate(tqdm(testloader)):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input, target)
            loss = criterion(output.squeeze(), target)                                  # Need to .squeeze() because of headed attention.
            pred = output.squeeze().argmax(dim=1)
            total_correct += (pred == target).sum().item()
        print(f'Validation Loss: {loss.item()}, Validation Accuracy: {total_correct/len(testloader.dataset)}')
    

if __name__ == "__main__":
    main()
