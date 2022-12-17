import logging
import sys 

import torch
import numpy as np
import matplotlib.pyplot as plt

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Optimizer():
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, valid_loader, device='cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.train_losses = []
        self.valid_losses = []
        
    def train_one_epoch(self, X, y):
        # train mode
        self.model.train()
        
        # zero grad
        self.optimizer.zero_grad()
        
        # makes prediction
        out = self.model(X)

        # compute sharpe ratio (loss)
        loss = self.loss_fn(out, y)

        # backpropagation
        # simulate gradient ascent with the gradient descent of -L
        (-loss).backward()
        self.optimizer.step()
        
        # return the loss
        return loss.item()
        
    def valid_one_epoch(self, X, y):       
        # makes prediction
        out = self.model(X)
        
        # compute sharpe ratio (loss)
        loss = self.loss_fn(out, y)
        
        return loss.item()
        
    def train(self, n_epochs=50, verbose=True):
        
        # epochs
        for epoch in range(1, n_epochs+1):
            
            # training 
            batch_losses = []
            
            for X_batch, y_batch in self.train_loader:
                # tensor to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # training step
                loss = self.train_one_epoch(X_batch, y_batch)
                
                # add loss
                batch_losses.append(loss)
            
            # take the mean of batches' loss
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)
            
            # update scheduler
            self.scheduler.step()
            
            # validation
            with torch.no_grad():
                batch_val_losses = []
                
                for X_batch, y_batch in self.valid_loader:
                    # to device
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    # validation step
                    loss = self.valid_one_epoch(X_batch, y_batch)
                    
                    # add loss
                    batch_val_losses.append(loss)
                    
                # take the mean of batches' loss
                valid_loss = np.mean(batch_val_losses)
                self.valid_losses.append(valid_loss)
                
            # show progression
            if (epoch <= 10 or epoch % 10 == 0) and verbose:
                print(f"[{epoch}/{n_epochs}] Train criterion : {(train_loss):>0.0000004f}, validation criterion : {(valid_loss):>0.0000004f}")
    
    def plot_losses(self):
        plt.title("Train and Validation criterion")
        plt.plot(self.train_losses, label='Training criterion')
        plt.plot(self.valid_losses, label='Validation criterion')
        plt.legend()
        plt.show()
        plt.close()