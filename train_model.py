import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer
import math
import os
from tqdm import tqdm
from safetensors.torch import save_file
import csv

# --- 1. Configuration (with added CHECKPOINT_PATH) ---
class Config:
    DATASET_PATH = "data/gec_dataset_from_c4.tsv"
    TOKENIZER_PATH = "gec_tokenizer.json"
    MODEL_SAVE_PATH = "model/gec_diffusion_model.safetensors"
    CHECKPOINT_PATH = "model/training_checkpoint.pth" # For resumable training

    D_MODEL = 256
    N_HEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 20 # Let's increase epochs, since we can resume
    DIFFUSION_TIMESTEPS = 1000

config = Config()

class GECDataset(Dataset):
    def __init__(self, tsv_path, tokenizer):
        self.tokenizer = tokenizer
        print("Loading dataset into memory...")
        self.data = pd.read_csv(tsv_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        incorrect_text = str(row['incorrect_sentence'])
        correct_text = str(row['correct_sentence'])
        incorrect_tokens = self.tokenizer.encode(incorrect_text).ids
        correct_tokens = self.tokenizer.encode(correct_text).ids
        return {'incorrect_ids': torch.tensor(incorrect_tokens, dtype=torch.long), 'correct_ids': torch.tensor(correct_tokens, dtype=torch.long)}

def collate_fn(batch, pad_id):
    incorrect_padded = nn.utils.rnn.pad_sequence([item['incorrect_ids'] for item in batch], batch_first=True, padding_value=pad_id)
    correct_padded = nn.utils.rnn.pad_sequence([item['correct_ids'] for item in batch], batch_first=True, padding_value=pad_id)
    return {'incorrect_ids': incorrect_padded, 'correct_ids': correct_padded}

class PositionalEncoding(nn.Module):
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
        if x.dim() == 3 and x.shape[1] > 1:
             x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GECDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(GECDiffusionModel, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.time_mlp = nn.Sequential(PositionalEncoding(d_model), nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, incorrect_src, corrupted_correct_tgt, time, pad_id):
        src_padding_mask = (incorrect_src == pad_id)
        tgt_padding_mask = (corrupted_correct_tgt == pad_id)
        src_embed = self.pos_encoder(self.token_embedding(incorrect_src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_embed, src_key_padding_mask=src_padding_mask)
        tgt_embed = self.pos_encoder(self.token_embedding(corrupted_correct_tgt) * math.sqrt(self.d_model))
        
        # Fix: Reshape `time` tensor to match expected dimensions
        time_embed = self.time_mlp(time.float().unsqueeze(-1))  # Add a dimension for compatibility
        
        tgt_embed += time_embed.unsqueeze(1)
        output = self.transformer_decoder(tgt_embed, memory, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.generator(output)

class DiffusionScheduler:
    def __init__(self, timesteps, tokenizer):
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.timesteps = timesteps
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def corrupt_sentence(self, x_start, t, device):
        mask_ratio = t.float() / self.timesteps
        random_values = torch.rand_like(x_start, dtype=torch.float32)
        should_mask = random_values < mask_ratio.unsqueeze(1)
        non_pad_mask = (x_start != self.pad_id)
        final_mask = should_mask & non_pad_mask
        corrupted = x_start.clone()
        corrupted[final_mask] = self.mask_id
        return corrupted

# --- Training Loop with Resumption Logic ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"--- Using device: {device} ---")

    tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    dataset = GECDataset(config.DATASET_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id))

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
    
    start_epoch = 0 # Default to starting from epoch 0

    # *** NEW: LOGIC TO LOAD CHECKPOINT ***
    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"--- Found checkpoint. Resuming training from {config.CHECKPOINT_PATH} ---")
        # Note: We load the checkpoint to the CPU first to avoid GPU memory issues, then move model to device.
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # Move model to the correct device
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # We start from the next epoch
        print(f"--- Resumed from Epoch {start_epoch}. Last recorded loss: {checkpoint['loss']:.4f} ---")
    else:
        print("--- No checkpoint found. Starting training from scratch. ---")

    print(f"--- Starting Training from Epoch {start_epoch+1} ---")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        scaler = torch.amp.GradScaler()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            optimizer.zero_grad()
            incorrect_ids, correct_ids = batch['incorrect_ids'].to(device), batch['correct_ids'].to(device)
            t = torch.randint(0, config.DIFFUSION_TIMESTEPS, (incorrect_ids.shape[0],), device=device)
            corrupted_correct = scheduler.corrupt_sentence(correct_ids, t, device)
            with torch.amp.autocast(device_type=device.type):
                output_logits = model(incorrect_ids, corrupted_correct, t, pad_token_id)
                loss = criterion(output_logits.view(-1, tokenizer.get_vocab_size()), correct_ids.view(-1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
                    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        # *** NEW: LOGIC TO SAVE CHECKPOINT AT THE END OF EACH EPOCH ***
        print("--- Saving training checkpoint... ---")
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
        torch.save(checkpoint, config.CHECKPOINT_PATH)
        print(f"--- Checkpoint saved to {config.CHECKPOINT_PATH} ---")
    
    # --- FINAL STEP: Save the clean model weights for inference ---

    print("\n--- Training Complete ---")
    print("Saving final model weights to .safetensors file...")
    save_file(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Final model for inference saved to: {config.MODEL_SAVE_PATH}")
