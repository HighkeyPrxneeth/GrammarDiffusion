import csv
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# --- Configuration ---
DATASET_FILE = "data/gec_dataset_from_c4.tsv"  # The large dataset you just created
TOKENIZER_FILE = "gec_tokenizer.json"    # The output file for the trained tokenizer

# 128k vocabulary size is a good starting point for many NLP tasks.
VOCAB_SIZE = 128000

# --- Data Iterator ---

def get_text_iterator(file_path: str):
    """
    Creates a memory-efficient iterator to read sentences from the large TSV file
    without loading the entire file into memory. It yields only the correct sentences,
    as we want our tokenizer's vocabulary to be based on clean, well-formed text.
    """
    print(f"Streaming sentences from {file_path} for tokenizer training...")
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            # The 'correct_sentence' is in the second column (index 1)
            if len(row) > 1:
                yield row[1]

# --- Main Training Logic ---

if __name__ == "__main__":
    # Instantiate a new tokenizer
    # We use WordPiece, which is used by BERT and is great for many languages.
    # It learns to break down unknown words into known sub-words.
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace() # Split by whitespace first

    # Define the trainer
    # The trainer will learn the sub-word vocabulary from our corpus.
    # These special tokens are essential for our diffusion model.
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[
            "[UNK]",  # Unknown token
            "[PAD]",  # Padding token
            "[SOS]",  # Start of Sentence
            "[EOS]",  # End of Sentence
            "[MASK]"  # The mask token for the diffusion process
        ]
    )

    # Train the tokenizer using the memory-efficient iterator
    print("Starting tokenizer training... This may take a while on a large dataset.")
    
    # Create the iterator from our dataset file
    text_iterator = get_text_iterator(DATASET_FILE)
    
    # The `train_from_iterator` method processes the data chunk by chunk.
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    
    # Save the trained tokenizer
    # to a single JSON file. This file is essential for both training and inference.
    tokenizer.save(TOKENIZER_FILE)

    print("\n--- Tokenizer Training Complete ---")
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")
    print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")