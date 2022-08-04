import numpy as np
from data.data_util import load_bundles, make_y, get_bundle_idx, get_multibundle_idx, get_bundle_for_idx
from utils.general_util import load_pickle
from evaluation import tsne_transform, umap_transform, pca_transform

class BundleData:
    
    def __init__(self, name, data_folder, **kwargs):
        
        self.name = name
        self.X_recon = None
        self.X_encoded = None
        self.X_encoded_tsne = None
        self.X_encoded_umap = None
        self.X_encoded_pca = None
        
        # Load streamlines
        if "parse_tract_func" in kwargs:
            X, bundle_idx = load_bundles(data_folder+name, **kwargs)
        else:
            X, bundle_idx = load_bundles(data_folder+name, self.parse_tract_name, **kwargs)
            
        self.X = X
        self.bundle_idx = bundle_idx # {bundle : [start, bundle_count]}

        # Get bundle labels
        y, bundle_num = make_y(self.bundle_idx) 
        self.y = y
        self.bundle_num = bundle_num # {index : bundle}
        
        print(f"Loaded {self.name} with {len(self.bundle_idx)} tracts and {len(self.X)} lines.")


    def get_subj_bundle_idx(self, bname):
        '''
            Get indices of bundle in X
            Example usage: subj.X[subj.get_subj_bundle_idx('V')]
        '''
        return get_bundle_idx(bname, self.bundle_idx)


    def get_subj_multibundle_idx(self, bundle_ls):
        '''
            Get indices of multiple bundle in X
            Example usage: subj.X[subj.get_subj_multibundle_idx(['CST_L','CST_R'])]
        '''
        return get_multibundle_idx(bundle_ls, self.bundle_idx)


    def get_subj_bundle_for_idx(self, line_idx):
        '''
            Retrieve bundle name given a streamline index
            Example usage: subj.get_subj_bundle_for_idx(32)
        '''
        return get_bundle_for_idx(line_idx, self.bundle_idx)
    
    @staticmethod
    def parse_tract_name(fname):
        '''
            Parse tract name from filename
            This method works for ADNI3 data, can also pass in custom function to __init__
            for other datasets with different naming formats.
        '''
        return "_".join(fname.split('_')[1:-2])
    
    
    def load_inference_data(self, fpath):
        '''
            Load inference data from file, include encoded embeddings and reconstruction
        '''
        result = load_pickle(fpath)
        self.X_encoded = result['X_encoded']
        self.X_recon = result['X_recon']


    def get_tsne_2d(self, perplexity=100, seed=0):
        '''Compute TSNE 2D from inferred embeddings'''
        if self.X_encoded_tsne is None and self.X_encoded is not None:
            self.X_encoded_tsne = tsne_transform(self.X_encoded, perplexity=perplexity,
                                                 tsne_dim=2, seed=seed)


    def get_umap_2d(self, seed=0):
        '''Compute UMAP 2D from inferred embeddings'''
        if self.X_encoded_umap is None and self.X_encoded is not None:
            self.X_encoded_umap = umap_transform(self.X_encoded, umap_dim=2, seed=seed)


    def get_pca_2d(self, seed=0, pca_from_file=None):
        '''Compute PCA 2D from inferred embeddings'''
        if self.X_encoded_pca is None and self.X_encoded is not None:
            self.X_encoded_pca = pca_transform(self.X_encoded, pca_dim=2, seed=seed,
                                               pca_from_file=pca_from_file)