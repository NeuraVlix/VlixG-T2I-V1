import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import string
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return hidden.squeeze(0)

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64*7*7, latent_dim)
        self.fc_var = nn.Linear(64*7*7, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64*7*7)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class VlixG1(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.image_decoder = ImageDecoder()
        self._build_vocab()
        
    def _build_vocab(self):
        chars = string.ascii_lowercase + string.digits + string.punctuation + ' '
        self.vocab = {c: i+1 for i, c in enumerate(chars)}
        self.vocab['<unk>'] = 0
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        
    def text_to_tensor(self, text):
        text = text.lower()
        tokens = [self.vocab.get(c, self.vocab['<unk>']) for c in text]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def encode(self, image, text):
        text_tensor = self.text_to_tensor(text)
        text_features = self.text_encoder(text_tensor)
        
        mu, log_var = self.image_encoder(image)
        combined = torch.cat([mu, text_features], dim=1)
        return combined, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, text):
        text_tensor = self.text_to_tensor(text)
        text_features = self.text_encoder(text_tensor)
        combined = torch.cat([z, text_features], dim=1)
        return self.image_decoder(combined)
    
    def forward(self, image, text):
        combined, mu, log_var = self.encode(image, text)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, text), mu, log_var
    
    def generate_from_text(self, text, device='cpu'):
        with torch.no_grad():
            z = torch.randn(1, 128).to(device)
            generated = self.decode(z, text)
            return generated.squeeze().cpu().numpy()