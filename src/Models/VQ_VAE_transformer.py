#Transformer VQ-VAE


#Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from vector_quantize_pytorch import VectorQuantize


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, n_heads=4, n_layers=3):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x, mask):
        #x: [B, P, F]
        x = self.input_proj(x)  #[B, P, D]

        #mask (invert True-False)
        attn_mask = ~mask  # inverti

        z = self.transformer(x, src_key_padding_mask=attn_mask)

        return z
    

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_heads=4, n_layers=3):
        super().__init__()

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers
        )

        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z, mask):
        attn_mask = ~mask

        z = self.transformer(z, src_key_padding_mask=attn_mask)
        x_recon = self.output_proj(z)

        return x_recon
    

class VQVAE_Transformer(L.LightningModule):
    def __init__(self, input_dim=3, latent_dim=128, codebook_size=256, n_heads=4, n_layers=3, dec=0.8, beta=0.25, rot_trick=True, lr=1e-3):
        
        super().__init__()

        self.save_hyperparameters()
        
        #Encoder
        self.encoder = TransformerEncoder(input_dim, latent_dim, n_heads, n_layers)
        
        #Decoder (not a "rea" transformer decoder)
        self.decoder = TransformerDecoder(latent_dim, input_dim, n_heads, n_layers)

        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=dec,
            commitment_weight=beta,
            rotation_trick=rot_trick
        )
        
        self.beta = beta
        self.lr = lr


    def forward(self, x, mask):

        z_e = self.encoder(x, mask)  # [B,P,D]

        z_q, indices, commit_loss = self.vq(z_e)

        x_recon = self.decoder(z_q, mask)

        commit_loss = (z_e - z_q) ** 2 
        commit_loss = (commit_loss * mask.unsqueeze(-1)).sum() / mask.unsqueeze(-1).sum()

        return x_recon, commit_loss, indices
    

    def training_step(self, batch, batch_idx):
        x, mask = batch

        x_recon, commit_loss, _ = self(x, mask)

        recon_loss = (x - x_recon) ** 2
        mask = mask.unsqueeze(-1)

        recon_loss = (recon_loss * mask).sum() / mask.sum()

        loss = recon_loss + (self.beta * commit_loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("commit_loss", commit_loss, prog_bar=True)

        return loss
    
    #Validation Step
    def validation_step(self, batch, batch_idx):

        x, mask = batch

        x_recon, commit_loss, _ = self(x, mask)
        
        #Reconstruction loss
        recon_loss = (x - x_recon) ** 2

        #Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        #Average only valid values
        recon_loss = recon_loss.sum() / mask.sum()
        
        #Total loss
        loss = recon_loss + commit_loss
        
        #Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
    

    #Test step
    def test_step(self, batch, batch_idx):
        x, mask = batch

        x_recon, commit_loss, _ = self(x, mask)

        #Reconstruction loss
        recon_loss = (x - x_recon) ** 2

        #Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        #Average only valid values
        recon_loss = recon_loss.sum() / mask.sum()

        #Total loss
        loss = recon_loss + commit_loss

        #Log
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss, prog_bar=True)
        self.log("test_commit_loss", commit_loss, prog_bar=True)
    
    #Optimizer
    def configure_optimizers(self):
        
        #Adam
        return torch.optim.Adam(self.parameters(), lr=self.lr)