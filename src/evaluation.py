import torch
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.general_util import set_seed, sample_idx, load_pickle
from data.data_util import make_data_loader


def get_reconstruction_vae(model, X, device='cpu', mean=None, std=None,
                           seed=0, split_batch_size=None):
    '''
        Get embedding and reconstruction for VAE for input data
        Used for inference
    '''
    model.to(device)
    model.eval()

    # normalize X
    X_norm = torch.from_numpy(X).to(device).sub(mean).div(std)
    
    # if input data is too big, use dataloader to split into batches
    if split_batch_size:
        eval_loader = make_data_loader(X_norm, seed, batch_size=split_batch_size, num_workers=0)
        print(f"Splitting into {len(eval_loader)} batches of {split_batch_size} for inference")
        
        X_encoded = torch.Tensor([]).to(device) # storing embedding
        X_recon = torch.Tensor([]).to(device) # storing reconstruction
        for inputs in eval_loader:
            with torch.no_grad():
                X_recon_batch, X_encoded_batch = model(inputs[0])
            X_encoded = torch.cat((X_encoded, X_encoded_batch), axis=0)
            X_recon = torch.cat((X_recon, X_recon_batch), axis=0)
            torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            X_recon, X_encoded = model(X_norm)
        
    X_encoded = X_encoded.cpu().detach().numpy()
    X_recon = X_recon.cpu().detach().numpy()

    # redo standardization
    if mean is not None and std is not None: 
        # print("Standardized...")
        X_recon = X_recon * std.cpu().detach().numpy() + mean.cpu().detach().numpy()

    # if not 3D data apply reshape
    if len(X_recon.shape) < 3:
        X_recon = X_recon.reshape((len(X_recon), -1, 3))

    return X_encoded, X_recon


def select_random_samples(model, X, n_sample=5, 
                        device="cpu", mean=None, std=None, seed=0):
    '''Select random sample from data and decode from model'''
    # set_seed(seed=seed)
    # idx_selected = np.random.choice(len(X), size=n_sample, replace=False)
    idx_selected = sample_idx(X, n_sample=n_sample, seed=seed)
    x = X[idx_selected]
    z, x_recon = get_reconstruction_vae(model, x, device=device, mean=mean, std=std)
    return x, z, x_recon


def decode_embeddings(z, model, device="cpu", mean=None, std=None):
    '''Decode embeddings using trained model'''

    model.to(device)
    model.eval()

    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).to(torch.float32)

    with torch.no_grad():
        x = model.decode(z.to(device)).cpu().detach().numpy()

    # apply standardization
    if mean is not None and std is not None:
        x = x * std.numpy() + mean.numpy()

    # if not 3D data apply reshape
    if len(x.shape) < 3:
        x = x.reshape((len(z), -1, 3))
    return x


def generate_random_samples(model, zdim = 2, n_sample=5, 
                            device='cpu', mean=None, std=None, seed=0):
    '''Generate random sample of ND embeddings and decode from model'''

    set_seed(seed=seed)
    z = torch.randn(n_sample, zdim).to(device)
    x = decode_embeddings(z, model, device=device, mean=mean, std=std)
    return x, z


def tsne_transform(X, tsne_dim=2, perplexity=100, seed=0):
    '''Apply TSNE on data'''
    tsne = TSNE(tsne_dim, perplexity=perplexity, 
                learning_rate='auto',
                init='pca', 
                random_state=seed)
    return tsne.fit_transform(X)


def umap_transform(X, umap_dim=2, seed=0):
    '''Apply UMAP on data'''
    reducer = umap.UMAP(n_components=umap_dim, random_state=seed)
    return reducer.fit_transform(X)


def pca_transform(X, pca_dim=2, seed=0, pca_from_file=None):
    '''Apply PCA on data'''
    if pca_from_file:
        pca = load_pickle(pca_from_file)
        return pca.transform(X)
    else:
        pca = PCA(n_components=pca_dim, random_state=seed)
        return pca.fit_transform(X)