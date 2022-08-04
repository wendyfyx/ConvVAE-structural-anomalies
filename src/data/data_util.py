import glob
import torch
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines, orient_by_streamline
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from utils.general_util import seed_worker, load_pickle


def load_streamlines(fpath, n_points=256, n_lines=None, 
                     align_streamline=None, preprocess=None, 
                     rng=None, verbose=True, **kwargs):

    '''
        Load streamlines from one .trk file (one tract).
        PARAMETERS:
            fpath       : file path of streamline (.trk) files.
            n_points    : number of points per streamline.
            n_line      : number of lines to load. If n_lines is larger than the number 
                        of streamlines in file, it loads all available streamlines.
            preprocess  : can be either 2D or 3D. For ConvVAE, use 3D (n_lines, n_points, 3).
            rng         : random state , e.p. np.random.RandomState(2022).
            verbose     : True or False.
        RETURNS:
            lines       : processed streamline bundle as np.array, return ArraySequence when 
                        n_points is not provided.
    '''

    lines = load_tractogram(fpath, reference="same", bbox_valid_check=False).streamlines
    if verbose:
        fname = fpath.split('/')[-1]
        print(f"Loaded {fname} with {len(lines)} lines each with {n_points} points.")

    if n_points is not None:
        lines = set_number_of_points(lines, n_points)
    if n_lines is not None:
        lines = select_random_set_of_streamlines(lines, n_lines, rng=rng)
    if align_streamline is not None:
        lines = orient_by_streamline(lines, align_streamline, n_points=n_points)

    if preprocess == "2d":
        lines = lines.get_data()
        if verbose:
            print(f"Preprocessed lines into {preprocess} with shape {lines.shape}")
    elif preprocess == "3d":
        if n_points is None:
            print("Cannot process into 3D if n_points=None, returning ArraySequence")
            return lines
        if n_lines is not None:
            n_lines = min(n_lines, len(lines))
        else:
            n_lines = len(lines)
        lines = lines.get_data().reshape((n_lines, n_points, 3))
        if verbose:
            print(f"Preprocessed lines into {preprocess} with shape {lines.shape}")

    return lines


def load_bundles(folder_path, parse_tract_func, min_lines=2, 
                align_bundles_path=None, tracts_exclude=None, 
                sub_folder_path="rec_bundles/", **kwargs):
    '''Load bundles in folder, sorted alphabetically by tract name. 
        PARAMETERS:
            folder_path         : the root folder path for each subject is (i.e. Subj01/).
            parse_tract_func    : a custom function that parse the file name to get the 
                                tract name (moved_CST_L__recognized.trk -> CST_L)
            min_lines           : minimum number of lines in a tract. Discard if below this
                                threshold.
            tracts_exclude      : a list containing tracts to not load.
            sub_folder_path     : (OPTIONAL) it's for when the bundle files are nested in other 
                                folders (Subj01/rec_bundles/*.trk).
            Can also pass in other arguments for load_streamlines above.
        RETURNS:
            lines               : Streamlines concatenated from all available bundles.
            bundle_idx          : dictionary, where the key is bundle name, and the value is a 
                                tuple (starting indexof bundle in lines, length of bundles).
    '''

    lines = []
    bundle_idx = {}

    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"
    if sub_folder_path:
        if not sub_folder_path.endswith("/"):
            sub_folder_path = sub_folder_path + "/"

    if align_bundles_path is not None:
        d_centroid = load_pickle(align_bundles_path)
        print(f"Loaded bundles to align with from {align_bundles_path}, {d_centroid['AF_L'].shape}")

    lines_count = 0
    for fpath in sorted(list(glob.glob(folder_path + sub_folder_path + "*.trk"))):  
        fname = fpath.split('/')[-1]
        tract = parse_tract_func(fname)
        
        if tracts_exclude:
            if tract in tracts_exclude:
                continue

        align_streamline = d_centroid[tract] if align_bundles_path is not None else None
        bundle = load_streamlines(fpath, align_streamline=align_streamline, **kwargs)
        if len(bundle) < min_lines:
            continue
            
        lines.append(bundle)
        bundle_idx[tract]=[lines_count, len(bundle)]
        lines_count += len(bundle)

    if len(lines)==0:
        print(f"No bundle was loaded from {folder_path}")
        return
    lines = np.concatenate(lines)

    return lines, bundle_idx


def make_y(bundle_idx):
    '''
        Make labels from bundle information. Return 1D array of index, and 
        a dictionary of the corresponding bundle names.
    '''
    y = []
    bundle_num = {}
    
    for idx, bundle in enumerate(sorted(bundle_idx)):
        y.append([idx] * bundle_idx[bundle][1])
        bundle_num[idx] = bundle
    y = np.concatenate(y)
    return y, bundle_num

def get_bundle_idx(bname, bundle_idx):
    '''
        Retreiving indices of bundle
    '''
    if bname not in bundle_idx:
        print(f"Bundle {bname} does not exist in this subject.")
        return np.arange(0)
    indices = bundle_idx[bname]
    return np.arange(indices[0], indices[0]+indices[1])


def get_multibundle_idx(bundle_ls, bundle_idx):
    '''
        Retreiving indices of multiple bundles
    '''
    all_idx = np.arange(0)
    for bname in bundle_ls:
        idx = get_bundle_idx(bname, bundle_idx)
        all_idx = np.concatenate([all_idx, idx])
    return all_idx


def get_bundle_for_idx(line_idx, bundle_idx):
    '''Retrieve bundle name given a streamline index'''
    for bname, v in bundle_idx.items():
        if line_idx >= v[0] and line_idx < v[0]+v[1]:
            return bname


def make_data_loader(data, seed, batch_size, num_workers=0):
    '''Make data loader to iterate through'''
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=g_seed)


def split_data(X, y=None, n_splits=10, test_size=0.2, random_state=1, verbose=True):
    '''
    [DATA-UTIL] Split data into train and test set
    PARAMETERS:
      n_splits: Number of re-shuffling & splitting iterations.
      test_size: Percentage of test data
    RETURNS:
      train: numpy array containing index of the train samples in df_x
      test: numpy array containing index of the test samples in df_x
    NOTE:
      To get train-val-test split, call split_data() once on orig data, once on train data
    '''
    X = [1 for _ in range(len(X))]
    if y is not None:
        sss = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
        train, test = next(sss.split(X, y))
    else:
        sss = ShuffleSplit(n_splits=n_splits,
                           test_size=test_size, random_state=random_state)
        train, test = next(sss.split(X))
    if verbose:
        print(f"Split into {len(train)} train and {len(test)} test samples," \
                f"with random state = {random_state}")
    return train, test


# def get_all_subj(data_folder):
#     '''[DATA-UTIL] Get a list of all subjects in a folder'''
#     return [subj.split('/')[-1] for subj in \
#                 sorted(list(glob.glob(f"{data_folder}/*")))]


def label_hemispheres(labels, suffix_l = '_L', suffix_r = '_R'):
    '''[DATA-UTIL] Label bundles by hemispheres according to bundle name suffix'''
    return ['Left' if x.endswith(suffix_l) else 'Right' if x.endswith(suffix_r) else 'Comm' for x in labels]


def sort_bundle_name_by_hemisphere(bundles, suffix_l = '_L', suffix_r = '_R'):
    '''
        [DATA-UTIL] Sort bundle from left - commissural - right
        PARAMETERS:
            bundles     : list of bundle names
            suffix_l    : optional parameter for left hemisphere bundle suffix
            suffix_r    : optional parameter for right hemisphere bundle suffix
        RETURNS:
            dict        : dictionary where key is bundle names in sorted order and new index
    '''
    hemisphere = [0 if x.endswith(suffix_l) else 2 if x.endswith(suffix_r) else 1 for x in bundles]
    return dict(zip([x for _, x in sorted(zip(hemisphere, bundles))], range(len(bundles))))