import numpy as np
from tqdm import tqdm
import torch

def optimize(model, criterion, optimizer, scheduler, trainloader, valloader, data_train, data_val, device, 
            epochs=100, patience=3, mname='simplenn'):
    # initialize logs
    logs = {'train': np.zeros(epochs), 'val': np.zeros(epochs)}

    for e in tqdm(range(epochs)):
        # fit model
        model.train()

        for inputs, target in tqdm(trainloader):
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad() # zero gradient
            
            # forward pass
            output = model(inputs)
            loss = criterion(output, target)
            
            # update epoch loss by weighted batch loss
            logs['train'][e] += loss.sum().item()

            # backwards pass
            loss.sum().backward()
            optimizer.step()

        logs['train'][e] /= len(data_train)
        print("Training loss: ", logs['train'][e])

        # evaluate model
        model.eval()

        with torch.no_grad():
            for inputs, target in tqdm(valloader):
                inputs, target = inputs.to(device), target.to(device)

                # forward pass
                output = model(inputs)
                loss = criterion(output, target)

                # update epoch loss by weighted batch loss
                logs['val'][e] += loss.sum().item()
                
        logs['val'][e] /= len(data_val)
        print("Validation loss: ", logs['val'][e])

        #scheduler.step()

        # save best model after each epoch
        if e > 0:
            if logs['val'][e] < np.amin(logs['val'][:e]):
                print("saving best model")
                torch.save(model, mname)

        # early stopping: break training loop if val loss increases for {patience} epochs
        if e > patience:
            if np.sum(np.diff(logs['val'])[e-patience:e] > 0) == [patience]:
                print("early stopping")
                return logs
    return logs