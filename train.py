#Main

#Import custom classes and libraries
from src.Data.data_loading import ParquetDataModule
from src.Data.data_loading import ParquetFeatureDataset
from src.Models.VQ_VAE_transformer import VQVAE_Transformer

import yaml
import numpy as np
from tqdm import tqdm
import torch
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import CSVLogger


def main():

    #Config
    with open("Configs/config.yaml") as f:
        config = yaml.safe_load(f)


    #Extract config info
    #Directories
    dirs_train = config["data"]["train_path"]
    dirs_val = config["data"]["val_path"]
    dirs_test = config["data"]["test_path"]

    #Features
    features_cols = config["data"]["features"]
    max_part = config["data"]["max_part"]
    prep = config["data"]["preprocessing"]

    #VQVAE hyperparameters
    input_dim = config["Transformer_VQVAE"]["input_dim"]
    latent_dim = config["Transformer_VQVAE"]["latent_dim"]
    codebook_size = config["Transformer_VQVAE"]["codebook_size"]
    rot_trick = config["Transformer_VQVAE"]["rotation_trick"]
    decay = config["Transformer_VQVAE"]["decay"]
    beta = config["Transformer_VQVAE"]["beta"]
    n_heads = config["Transformer_VQVAE"]["n_heads"]
    n_layers = config["Transformer_VQVAE"]["n_layers"]

    #Training hyperparameters
    lr = config["training"]["lr"]
    max_epochs = config["training"]["max_epochs"]
    batch_size = config["training"]["batch_size"]


    #Initialization of itarable dataset (train dataset) containing the constituents' features (eta, phi, pT)
    #In particular I am currently using the first 2 parquet files
    dataset_train = ParquetFeatureDataset(
        parquet_dirs=dirs_train,
        features=features_cols,
        max_particles=max_part,
        batch_size=batch_size,
        preprocessing=prep
    )

    #Validation dataset
    #In particular I am currently using parquet files n. 5,6
    dataset_val = ParquetFeatureDataset(
        parquet_dirs=dirs_val,
        features=features_cols,
        max_particles=max_part,
        batch_size=batch_size,
        preprocessing=prep
    )

    #Test dataset
    #In particular I am currently using parquet file n. 7
    dataset_test = ParquetFeatureDataset(
        parquet_dirs=dirs_test,
        features=features_cols,
        max_particles=max_part,
        batch_size=batch_size,
        preprocessing=prep
    )

    #Initialization of the lightining datamodule 
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

    #Model
    model_with_rot = VQVAE_Transformer(
        dec=decay,
        beta = beta,
        rot_trick=rot_trick,
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        codebook_size=codebook_size,
        lr=lr
    )
    
    #Logger
    logger = CSVLogger("logs", name="transf_vqvae")
    
    #Trainer 
    trainer_with_rot = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",   #CPU/GPU automatic
        devices="auto",
        log_every_n_steps=10,
        logger=logger
    )

    #Training
    trainer_with_rot.fit(model_with_rot, datamodule=data_module)

    #Test
    trainer_with_rot.test(model_with_rot, datamodule=data_module)


if __name__ == "__main__":
    main()