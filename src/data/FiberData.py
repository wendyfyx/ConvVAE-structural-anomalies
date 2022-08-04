import numpy as np
from sklearn.preprocessing import LabelEncoder
from data.BundleData import BundleData
from data.data_util import get_bundle_idx
from utils.general_util import map_list_with_dict, load_pickle

class FiberData:
    
    '''
        Used for loading multiple subjects' for training
        subjs, X, y, subj_idx, bundle_num, subj_bundle_idx
    '''
    
    def __init__(self, subjs, **kwargs):
        self.subjs = subjs
        self.subj_idx = {} # keep track of subject index
        self.subj_bundle_idx = {} # keep track of bundle indices for each subject
        self.X = None
        self.y = None
        self.X_encoded = None
        self.X_recon = None
        self.bundle_num = None
        self.load_subjects(**kwargs)
    

    def load_subjects(self, **kwargs):
        
        # Load first subject and train a label encoder
        subj0 = BundleData(self.subjs[0], **kwargs)
        le = LabelEncoder()
        le.fit(map_list_with_dict(subj0.y, subj0.bundle_num))
        
        X = subj0.X
        y = subj0.y
        
        self.subj_idx[self.subjs[0]] = [0, len(subj0.X)]
        self.subj_bundle_idx[self.subjs[0]] = subj0.bundle_idx
        
        # Load other subjects
        lines_count = len(subj0.X)
        for subj in self.subjs[1:]:
            subj_data = BundleData(subj, **kwargs)
            self.subj_bundle_idx[subj] = subj_data.bundle_idx
            
            y_transformed = le.transform(map_list_with_dict(subj_data.y, 
                                                            subj_data.bundle_num))
            X = np.vstack((X, subj_data.X))
            y = np.hstack((y, y_transformed))
            
            self.subj_idx[subj] = [lines_count, len(subj_data.X)]
            lines_count += len(subj_data.X)
            
        # Save variables
        self.X = X
        self.y = y
        self.bundle_num = subj0.bundle_num
        print(f"Loaded {len(self.subjs)} subjects, with a total of {len(self.X)}")
        

    def get_subj_idx(self, subj):
        '''
            Get indices of all bundles subject in X
        '''
        if subj not in self.subjs:
            print(f"Subject {subj} does not exist.")
            return
        indices = self.subj_idx[subj]
        return np.arange(indices[0], indices[0]+indices[1])
    

    def get_bundle_for_subj(self, subj, bundle):
        '''
            Get specific bundle index for subject
        '''
        subj_idx = self.get_subj_idx(subj)
        bundle_idx = get_bundle_idx(bundle, self.subj_bundle_idx[subj])
        return subj_idx[bundle_idx]

    
    def load_inference_data(self, model_path, epoch):
        '''Load inference data for all subjects'''
        X_encoded = []
        X_recon = []
        for subj in self.subjs:
            subj_result = load_pickle(f"{model_path}/E{epoch}_{subj}")
            X_encoded.append(subj_result['X_encoded'])
            X_recon.append(subj_result['X_recon'])
        self.X_encoded = np.vstack(X_encoded)
        self.X_recon = np.vstack(X_recon)


    def split_train_test(self, **kwargs):
        # TODO
        pass
