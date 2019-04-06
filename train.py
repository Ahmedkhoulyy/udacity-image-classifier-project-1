import torch
import utils
import model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help='Directory of the files')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Directory to save the model')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg19',
                    help='Architecture of the model')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.0008,
                    help='Learning rate of the model')
parser.add_argument('--hidden_units', action='append',
                    dest='hidden_units',
                    default=[1024, 512],
                    help='Hidden units of the model')
parser.add_argument('--ephocs', action='store',
                    dest='epochs',
                    default=5,
                    help='Epochs to train the model')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set training to gpu')

results = parser.parse_args()

device = ("cuda" if results.gpu else "cpu")
dataloaders, dataset_sizes = utils.preprocessing_images(results.data_dir)
model, criterion, optimizer = model.model(hidden_layers=results.hidden_units, 
                                          learning_rate=results.learning_rate, 
                                          arch=results.arch, 
                                          device=device)

# Training the model
epochs = results.epochs
steps = 0
running_loss = 0

training_losses, validation_losses = [], []

for e in range(epochs):
    for inputs, targets in dataloaders['train']:
        steps += 1
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)                               
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                valid_loss += batch_loss.item()
                
                # Accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {e+1}/{epochs} (steps: {steps}).. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}.. ")
        
        training_losses.append(running_loss)
        validation_losses.append(valid_loss)
        running_loss = 0
        steps = 0
        model.train()
        
checkpoint = {'epochs': epochs,
              'training_losses': training_losses,
              'validation_losses': validation_losses,
              'layers': results.hidden_units,
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'model_checkpoint.pth')