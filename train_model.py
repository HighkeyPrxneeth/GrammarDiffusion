import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer
import math
from tqdm import tqdm
from safetensors.torch import save_file
import csv

# --- 1. Configuration ---
# These hyperparameters are a good starting point. You can tune them for better performance.
class Config:
    # --- File Paths ---
    DATASET_PATH = "data/gec_dataset_from_c4.tsv"
    TOKENIZER_PATH = "gec_tokenizer.json"
    MODEL_SAVE_PATH = "model/gec_diffusion_model.safetensors"

    # --- Model Dimensions ---
    D_MODEL = 256
    N_HEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # --- Training Parameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 10

    # --- Diffusion Parameters ---
    DIFFUSION_TIMESTEPS = 1000
    
# Instantiate config
config = Config()

# --- 2. Dataset and Dataloader ---

class GECDataset(Dataset):
    """
    Custom PyTorch Dataset to load the GEC data.
    It reads the TSV file and tokenizes the sentences on the fly.
    """
    def __init__(self, tsv_path, tokenizer):
        self.tokenizer = tokenizer
        # To avoid loading the entire large file into memory, we'll just store the file path
        # and read it line by line in __getitem__. For very large datasets, this can be slow.
        # A better approach for production is to pre-tokenize and save as a binary format.
        # But for this project, reading line-by-line is a reasonable tradeoff.
        print("Loading dataset into memory. For extremely large datasets, consider pre-tokenizing.")
        self.data = pd.read_csv(tsv_path, sep='\t', quoting=csv.QUOTE_MINIMAL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        incorrect_text = str(row['incorrect_sentence'])
        correct_text = str(row['correct_sentence'])

        # Tokenize texts using the trained tokenizer
        incorrect_tokens = self.tokenizer.encode(incorrect_text).ids
        correct_tokens = self.tokenizer.encode(correct_text).ids
        
        return {
            'incorrect_ids': torch.tensor(incorrect_tokens, dtype=torch.long),
            'correct_ids': torch.tensor(correct_tokens, dtype=torch.long)
        }

def collate(batch, pad_id):
    """
    A custom collate function to pad sequences in each batch to the same length.
    This is necessary for batching variable-length sentences.
    """
    incorrect_padded = nn.utils.rnn.pad_sequence(
        [item['incorrect_ids'] for item in batch], batch_first=True, padding_value=pad_id
    )
    correct_padded = nn.utils.rnn.pad_sequence(
        [item['correct_ids'] for item in batch], batch_first=True, padding_value=pad_id
    )
    
    return {
        'incorrect_ids': incorrect_padded,
        'correct_ids': correct_padded,
    }

# --- 3. Model Architecture ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
        if x.dim() == 3 and x.shape[1] > 1: # Batch first
             x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else: # Seq first
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GECDiffusionModel(nn.Module):
    """
    The main Transformer-based Encoder-Decoder model for GEC diffusion.
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(GECDiffusionModel, self).__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # A simple MLP to embed the timestep `t` into a vector of size d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model * 4),
            nn.Mish(), # A smooth activation function
            nn.Linear(d_model * 4, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, incorrect_src, corrupted_correct_tgt, time, pad_id):
        # Create masks for padding tokens
        src_padding_mask = (incorrect_src == pad_id)
        tgt_padding_mask = (corrupted_correct_tgt == pad_id)

        # 1. Encode the incorrect sentence (the condition)
        src_embed = self.pos_encoder(self.token_embedding(incorrect_src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_embed, src_key_padding_mask=src_padding_mask)

        # 2. Prepare the decoder input (corrupted sentence + time embedding)
        tgt_embed = self.pos_encoder(self.token_embedding(corrupted_correct_tgt) * math.sqrt(self.d_model))
        time_embed = self.time_mlp(time.float().unsqueeze(-1))
        
        # Add the time embedding to each token embedding in the target sequence
        # This informs the decoder about the current noise level.
        tgt_embed += time_embed.unsqueeze(1)
        
        # 3. Decode to predict the original sentence
        # The decoder uses its self-attention on the corrupted target and cross-attention on the encoder's output.
        output = self.transformer_decoder(tgt_embed, memory, 
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=src_padding_mask)
        
        return self.generator(output)

# --- 4. Diffusion Logic (Scheduler) ---

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionScheduler:
    def __init__(self, timesteps, tokenizer):
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.timesteps = timesteps
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def corrupt_sentence(self, x_start, t, device):
        """
        Applies masking-based corruption to a batch of sentences.
        This is a simplified diffusion process where "noise" means replacing tokens with [MASK].
        """
        n_steps = self.alphas_cumprod.to(device)
        
        # Calculate the probability of a token being masked at timestep t
        # We use a simple linear mapping from t to probability
        mask_ratio = t.float() / self.timesteps
        
        # Create a random mask tensor
        # We need to reshape mask_ratio to (batch_size, 1) to compare with random_values
        random_values = torch.rand_like(x_start, dtype=torch.float32)
        should_mask = random_values < mask_ratio.unsqueeze(1)
        
        # Do not mask padding tokens
        non_pad_mask = (x_start != self.pad_id)
        final_mask = should_mask & non_pad_mask

        corrupted = x_start.clone()
        corrupted[final_mask] = self.mask_id
        
        return corrupted

# --- 5. Training Loop ---

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # Load tokenizer and create dataloader
    tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    dataset = GECDataset(config.DATASET_PATH, tokenizer)
    # We create the collate_fn with a lambda to pass the pad_id
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                            collate_fn=lambda b: collate(b, pad_token_id))

    # Initialize model, optimizer, scheduler, and loss function
    model = GECDiffusionModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = DiffusionScheduler(config.DIFFUSION_TIMESTEPS, tokenizer)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    print("--- Starting Training ---")
    print(f"Dataset Size: {len(dataset)}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            optimizer.zero_grad()
            
            incorrect_ids = batch['incorrect_ids'].to(device)
            correct_ids = batch['correct_ids'].to(device)
            
            # 1. Sample a random timestep `t` for each sentence in the batch
            t = torch.randint(0, config.DIFFUSION_TIMESTEPS, (incorrect_ids.shape[0],), device=device)
            
            # 2. Corrupt the 'correct' sentences to create the decoder input
            corrupted_correct = scheduler.corrupt_sentence(correct_ids, t, device)
            
            # 3. Forward pass
            output_logits = model(incorrect_ids, corrupted_correct, t, pad_token_id)
            
            # 4. Calculate loss
            # The model must predict the original 'correct_ids' from the corrupted version.
            # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize)
            loss = criterion(output_logits.view(-1, tokenizer.get_vocab_size()), correct_ids.view(-1))
            
            # 5. Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

    # --- 6. Save the Final Model Weights ---
    print("\n--- Training Complete ---")
    print("Saving model weights to .safetensors file...")
    save_file(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved successfully to: {config.MODEL_SAVE_PATH}")
