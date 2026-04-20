#MLP VQ-VAE

#Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from vector_quantize_pytorch import VectorQuantize


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # x: [B, P, F]
        B, P, F = x.shape
        x = x.view(B * P, F)
        z = self.net(x)
        z = z.view(B, P, -1)
        return z


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=128, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        # z: [B, P, D]
        B, P, D = z.shape
        z = z.view(B * P, D)
        x_recon = self.net(z)
        x_recon = x_recon.view(B, P, -1)
        return x_recon


# VQ-VAE Model (Lightning)
class VQVAE_MLP(L.LightningModule):
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=256, codebook_size=256, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()

        # Encoder / Decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

        # Vector Quantizer
        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            rotation_trick=True,
            commitment_weight=0.25
        )

        self.lr = lr

    
    #Forward
    def forward(self, x):
        # x: [B, P, F]

        z_e = self.encoder(x)                     # [B, P, D]
        z_q, indices, commit_loss = self.vq(z_e)  # quantization
        x_recon = self.decoder(z_q)               # reconstruction

        return x_recon, commit_loss, indices

    #Training Step
    def training_step(self, batch, batch_idx):
        x, mask = batch  # mask: [B, P]

        x_recon, commit_loss, _ = self(x)

        #Reconstruction loss (MSE)
        recon_loss = (x - x_recon) ** 2

        #Apply mask
        mask = mask.unsqueeze(-1)  # [B, P, 1]
        recon_loss = recon_loss * mask

        #Avarage only with valid values
        recon_loss = recon_loss.sum() / mask.sum()

        #Total loss
        loss = recon_loss + commit_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("commit_loss", commit_loss, prog_bar=True)

        return loss

    #Validation Step
    def validation_step(self, batch, batch_idx):
        x, mask = batch

        x_recon, commit_loss, _ = self(x)

        recon_loss = (x - x_recon) ** 2
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask
        recon_loss = recon_loss.sum() / mask.sum()

        loss = recon_loss + commit_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
    
    #Test step
    def test_step(self, batch, batch_idx):
        x, mask = batch

        x_recon, commit_loss, _ = self(x)

        # Reconstruction loss
        recon_loss = (x - x_recon) ** 2

        # Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        # Average only valid values
        recon_loss = recon_loss.sum() / mask.sum()

        # Total loss
        loss = recon_loss + commit_loss

        # Log
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss, prog_bar=True)
        self.log("test_commit_loss", commit_loss, prog_bar=True)

    #Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)