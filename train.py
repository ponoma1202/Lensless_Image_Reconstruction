import torch
from model import Transformer
import torchvision

def train():
    print('Training model...')

def main():
    # Use CIFAR 10 for the data
    batch_size = 4
    num_epochs = 10
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),        # from https://github.com/kuangliu/pytorch-cifar/issues/19  
                                                torchvision.transforms.Grayscale()])             
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    in_dim = 32*32           # Dimension of input sequence
    out_dim = 10          # Dimension of output sequence. For classification = number of classes
    
    # Initialize model
    model = Transformer(in_dim, out_dim)
    
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_device()

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
            output = model(input)                                                       # result is a (num_classes, batch_size) tensor
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
