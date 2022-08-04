# ConvVAE for Detecting Structural Anomalies in White Matter Tracts

This repository contains code Jupyter notebooks for paper  *Learning Optimal White Matter Tract Representations from Tractography using a Deep Generative Model for Population Analyses* ([bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.31.502227v1)).

---

### Running the Models

Two environment file (`environment-cpu.yml` and `environment-cuda.yml` ) are provided. Run `conda env create -f environment-[device].yml` to create the appropriate environment.

---

### Notebooks `/notebooks`

- `ConvVAE-train.ipynb` contains code for loading data and training `ConvVAE` model.
- `ConvVAE-inference.ipynb` contains code for inference on new subjects.
- `ConvVAE-eval.ipynb` contains code for plotting embeddings and distance preservation analysis (along with code for generating plots).
- `Anomaly-Detection.ipynb ` contains code for conducting structural anomly detection.

---

### Result files `/results`

Run `setup.sh` to create the folders. All data saved from model training/inference is described below.

- `/data` contains:
	- Embeddings (`X_encoded`), reconstructed streamlines (`X_recon`) and TSNE transformed data in the format of `[model_name]/E[epoch]_[subject].pkl` where each trained model has its own subfolder.
- `/models` are pytorch models, in the format of `[model_class]_[Conv_initialization+Linear_initialization]_Z[embedding_dim]_B[batch_size]_LR[learning_rate]_WD[weight_decay]_GC(V/N)[gradient_clip]_E[epochs_trained]_[subject_setting].` 
	- Model `state_dict`, last epoch and batch number for which the model was saved (used to resume training if needed), training data mean and std are saved. 
	- <u>Initialization</u>: XU for Xavier uniform, XN for Xavier normal, KU for Kaiming uniform, and KN for Kaiming normal; all bias are set to zero regardless of which initialization.
	- <u>Gradient</u> clip: GCV for clip gradient value (specifies the `clip_value` parameter in the pytorch function), and GCN for clip gradient norm (default L2 norm, specifies the `max_norm` parameter in the pytorch function)
	- Note that multiple epochs for each model are saved, these can all be loaded for inference.
	- Use `parse_model_setting(model_name)` to get the hyperparameters as `dict`.
- `/logs` contains tensorboard log files (for training and eval loss). Each model has their own folder, so we can compare different loss plots.
