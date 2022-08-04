import torch
import argparse
import numpy as np

from data.BundleData import BundleData
from evaluation import *
from utils.general_util import *
from data.data_util import *
from model.model import *

def apply_inference(model, subj_data, 
                    mean, std, device,
                    model_name, epoch,
                    seed=0, split_batch_size=None,
                    result_data_folder="../results/data/",
                    save_result=True):
    result = {}
    
    X_encoded, X_recon = get_reconstruction_vae(model, subj_data.X,
                                                device=device, mean=mean, std=std,
                                                seed=seed,
                                                split_batch_size=split_batch_size)
    result["X_encoded"] = X_encoded
    result["X_recon"] = X_recon
        
    # if perplexity:
    #     print("Applying TSNE...")
    #     result[f"X_encoded_tsne"] = tsne_transform(X_encoded, perplexity=perplexity, seed=seed)
    
    if save_result:
        if not result_data_folder.endswith("/"):
            result_data_folder = result_data_folder + "/"
        result_fpath = f"{result_data_folder}{model_name}/E{epoch}_{subj_data.name}"
        print(f"Saving result to {result_fpath}")
        save_pickle(result, result_fpath)
    
    return result


def load_model_for_inference(name, model_folder, epoch, device, model_type="checkpoint"):
    model_info = parse_model_setting(name)
    model = init_new_model(model_type=model_info['model_type'], 
                        Z=model_info['Z'])

    if model_type == "inference":
        model.load_state_dict(torch.load(f"{model_folder}{name}/model_{model_type}_E{epoch}", 
                                        map_location=torch.device(device)))
    elif model_type == "checkpoint":
        checkpoint = torch.load(f"{model_folder}{name}/model_{model_type}_E{epoch}", 
                                map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['mean'], checkpoint['std']


def apply_inference_main(args):

    if args.device == "cuda" and args.device_num:
        torch.cuda.set_device(args.device_num)
        print(f"Set cuda device to {args.device_num}")
    
    # Load model
    model, mean, std = load_model_for_inference(args.model_name, args.model_folder, args.epoch, 
                    args.device, args.model_type)

    # Define required params
    data_args = {'n_points' : 256, 'n_lines' : None, 'min_lines' : 2, 
            'tracts_exclude' : ['CST_L_s', 'CST_R_s'], 'preprocess' : '3d', 
            'rng' : np.random.RandomState(args.seed), 'verbose': False, 
            'align_bundles_path' : args.align_bundle_fpath, 
            'data_folder' : args.data_folder}

    # load subject data
    for subj in args.subj_list:
        subj_data = BundleData(subj, **data_args)

        # # apply mean and std to new data
        # subj_data.X_norm = torch.from_numpy(subj_data.X).to(args.device).sub(mean).div(std)

        # apply inference
        apply_inference(model, subj_data, 
                        mean, std, 
                        device=args.device,
                        model_name=args.model_name, 
                        epoch=args.epoch,
                        seed=args.seed,
                        split_batch_size=args.split_batch_size,
                        result_data_folder=args.result_data_folder,
                        # perplexity=args.perplexity,
                        save_result=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--subj_list', nargs='+', required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--split_batch_size', type=int, required=False)
    parser.add_argument('--device_num', type=int, required=False)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--align_bundle_fpath', type=str, required=True)
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--result_data_folder', type=str, required=True)

    args = parser.parse_args()
    apply_inference_main(args)

if __name__ == '__main__':
    main()

