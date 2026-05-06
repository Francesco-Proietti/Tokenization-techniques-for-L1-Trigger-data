#Import custom classes and libraries
from src.Data.data_loading import ParquetDataModule
from src.Data.data_loading import ParquetFeatureDataset
from src.Models.VQ_VAE_MLP import VQVAE_MLP 

import yaml
import numpy as np
from tqdm import tqdm
import torch
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import CSVLogger


def main():

    #Open config file---------------------------------------------
    with open("Configs/config.yaml") as f:
        config = yaml.safe_load(f)


    #Extract config info------------------------------------------
    #Directories
    dirs_train = config["data"]["train_path"]
    dirs_val = config["data"]["val_path"]
    dirs_test = config["data"]["test_path"]

    #Features
    features_cols = config["data"]["features"]
    max_part = config["data"]["max_part"]
    prep = config["data"]["preprocessing"]

    #VQVAE hyperparameters
    input_dim = config["MLP_VQVAE"]["input_dim"]
    hidden_dim = config["MLP_VQVAE"]["hidden_dim"]
    latent_dim = config["MLP_VQVAE"]["latent_dim"]
    codebook_size = config["MLP_VQVAE"]["codebook_size"]
    rot_trick = config["MLP_VQVAE"]["rotation_trick"]
    decay = config["MLP_VQVAE"]["decay"]
    beta = config["MLP_VQVAE"]["beta"]

    #Training hyperparameters
    lr = config["training"]["lr"]
    max_epochs = config["training"]["max_epochs"]
    batch_size = config["training"]["batch_size"]


    #Initialization of the lightining datamodule------------------
    data_module = ParquetDataModule(
        parquet_dirs_train=dirs_train, 
        parquet_dirs_val=dirs_val,
        parquet_dirs_test=dirs_test,
        features=features_cols,
        window_particles=max_part,
        batch_size=batch_size,
        preprocessing=prep
        #num_workers=0
    )


    #Model--------------------------------------------------------
    model_with_rot = VQVAE_MLP(
        dec=decay,
        beta = beta,
        rot_trick=rot_trick,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        lr=lr
    )

    #Logger
    logger = CSVLogger("logs", name="mlp_vqvae")

    #Trainer (Lightning)
    trainer_with_rot = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",   #CPU/GPU automatic
        devices="auto",
        log_every_n_steps=10
    )


    #Training-----------------------------------------------------
    trainer_with_rot.fit(model_with_rot, datamodule=data_module)


    #Test---------------------------------------------------------
    trainer_with_rot.test(model_with_rot, datamodule=data_module)


if __name__ == "__main__":
    main()