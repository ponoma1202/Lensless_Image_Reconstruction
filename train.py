import torch
from model import Transformer
import torchvision
from utils import Rescale

def train():
    print('Training model...')

def main():
    # Use CIFAR 10 for the data
    batch_size = 4
    num_epochs = 10
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
    
    # Initialize model
    model = Transformer(in_dim, out_dim)
    
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)           # TODO: potentially add learning rate scheduler later and change parameters to paper's params
    criterion = torch.nn.CrossEntropyLoss()

    train(model, trainloader, optimizer, criterion, num_epochs, device)
    validate(model, testloader, criterion, device)


def train(model, trainloader, optimizer, criterion, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        total_correct = 0
        for step, batch in enumerate(trainloader):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input, target)                                                       # result is a (num_classes, batch_size) tensor
            optimizer.zero_grad()
            loss = criterion(output, target)                                            # take argmax to get the class with the highest "probability"
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()                              # sum for list is a list so need to use .item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {total_correct/len(trainloader.dataset)}')

def validate(model, testloader, criterion, device):
    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(testloader):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()
        print(f'Validation Loss: {loss.item()}, Validation Accuracy: {total_correct/len(testloader.dataset)}')
    

if __name__ == "__main__":
    main()
