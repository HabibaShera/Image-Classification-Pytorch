import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.pooling import MaxPool2d

class IntelCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
    
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),  
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(1024*8*8, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64,6)
    )

  def forward(self, image):
    output = self.model(image)
    return output



def accuracy(pred, label):
  """
  This function calculate the accuracy. It takes:
  pred  : predictions of the model
  label : true labels for images

  return accuracy-->(float)
  """
  max_value , out_indices = torch.max(pred, dim=1)
  return torch.tensor(torch.sum(out_indices==label).item()/len(pred))

def validation_step(valid_dl, model, loss_fn):
  """
  calculate the accuracy and loss on the validation data. It takes:
  valid_dl : validation data loader
  model    : the model
  loss_fn  : loss function

  return {val_loss, val_acc}-->dict
  """
  for image, label in valid_dl:
    output = model(image)
    loss = loss_fn(output, label)
    acc = accuracy(output, label)
    return {'val_loss':loss, "val_acc":acc}

def fit(train_dl, valid_dl, epochs, optimizer, loss_fn, model):
  """
  training function. It takes:
  train_dl  : train dataloader
  valid_dl  : valid dataloader
  epochs    : number of epochs
  optimizer : model's optimizer
  loss_fn   : loss function
  model     : model

  return history of the model which contains (loss, val_loss and val_acc)-->list
  """
  history = []
  for epoch in range(1, epochs+1):
    for image,label in train_dl:
      output = model(image)
      loss = loss_fn(output, label)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    val = validation_step(valid_dl, model, loss_fn)
    print(f"Epoch [{epoch}/{epochs}] ==> loss: {loss}, val_loss:{val['val_loss']}, val_acc: {val['val_acc']}")
    history.append({'loss':loss,
                    'val_loss': val['val_loss'],
                    'val_acc':val['val_acc']
                    })
  return history

def to_device(data, device):
  """
  save data to available device such cpu or gpu. It takes:
  data   : list, tuple, data loader
  device : takes 'cpu' or 'gpu'
  """
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device

  def __iter__(self):
    for x in self.dl:
      yield to_device(x, self.device)