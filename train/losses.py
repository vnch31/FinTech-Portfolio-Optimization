import torch
import numpy as np

def sharpe_ratio_loss(weights, y, device='cpu'):
    # reshape weights
    weights = torch.unsqueeze(weights, 1)
    print(weights)
    
    # compute mean returns
    last_return = torch.unsqueeze(y[:, -1, :], 2)
    print(last_return)
    
    # compute returns
    portfolio_returns = torch.matmul(weights, last_return)
    
    # compute mean and std
    std, mean = torch.std_mean(portfolio_returns)

    return (mean/std)

def sortino_ratio_loss(weights, y, device='cpu'):
    # reshape weights
    weights = torch.unsqueeze(weights, 1)
    
    # compute mean returns
    last_return = torch.unsqueeze(y[:, -1, :], 2)
    
    # compute returns
    portfolio_returns = torch.matmul(weights, last_return)
    
    # compute mean
    mean = torch.mean(portfolio_returns)

    # negative std
    neg_returns = torch.FloatTensor([ret for ret in portfolio_returns if ret < 0])
    std = torch.mean(neg_returns)
    
    return (mean/std)

def sharpe_ratio_loss_github(weights, y, device='cpu'):
    # reshape weights
    weights = torch.unsqueeze(weights, 1)
    
    # compute mean returns
    mean_return = torch.unsqueeze(torch.mean(y, axis=1), 2)
    
    # cov matrix 
    covmat = torch.Tensor(np.array([np.cov(batch.cpu().T, ddof=0) for batch in y])).to(device)
    
    # compute portfolio returns
    portfolio_returns = torch.matmul(weights, mean_return)
    
    # compute portfolio volatility
    portfolio_vol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    
    # derive sharpe ratio
    sharpe_ratio = (portfolio_returns * 252 - 0.02) / (torch.sqrt(portfolio_vol * 252))
    #sharpe_ratio = portfolio_returns / torch.sqrt(portfolio_vol)
    
    return sharpe_ratio.mean()