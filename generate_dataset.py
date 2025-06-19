import csv
import random
import re
from typing import List

# Import libraries for data handling
from datasets import load_dataset
from tqdm import tqdm
import nltk

# --- Configuration ---
# Make sure you have run nltk.download('punkt') once before
try:
    nltk.data.find('tokenizers/punkt')
except nltk.data.LookupError:
    print("NLTK 'punkt' model not found.")
    nltk.download('punkt')
    nltk.data.find('tokenizers/punkt')


# --- Re-usable Corruption Engine Class ---
class CorruptionEngine:
    """
    A comprehensive engine to programmatically introduce a wide variety of grammatical
    and spelling errors into a sentence.
    """

    def __init__(self):
        # (Incorrect -> Correct) mapping for swapping
        self.homophones = {
            "your": "you're", "you're": "your", "their": "there", "there": "their",
            "they're": "their", "its": "it's", "it's": "its", "to": "too",
            "too": "to", "two": "to", "affect": "effect", "effect": "affect",
            "then": "than", "than": "then", "which": "witch", "weather": "whether",
            "break": "brake", "buy": "by", "hear": "here", "hole": "whole",
            "passed": "past", "peace": "piece", "principal": "principle"
        }

        self.articles = {"a", "an", "the"}
        self.prepositions = {"of", "in", "on", "for", "with", "at", "by", "from", "about"}
        self.aux_verbs = {"is", "am", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
                          "will", "would", "shall", "should", "may", "might", "must", "can", "could"}

        # A list of all possible corruption methods for random selection.
        self.corruption_methods = [
            self.replace_with_homophone,
            self.replace_with_typo,
            self.corrupt_subject_verb_agreement,
            self.delete_word_type,
            self.delete_punctuation,
            self.add_redundant_word,
            self.add_unnecessary_punctuation,
            self.swap_adjacent_words
        ]

    def replace_with_homophone(self, words: List[str]) -> List[str]:
        possible_indices = [i for i, word in enumerate(words) if word in self.homophones]
        if possible_indices:
            idx = random.choice(possible_indices)
            words[idx] = self.homophones[words[idx]]
        return words

    def replace_with_typo(self, words: List[str]) -> List[str]:
        if not words: return words
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        if len(word) < 3: return words
        char_idx = random.randint(0, len(word) - 1)
        typo_type = random.choice(['swap', 'insert', 'delete', 'replace'])
        chars = list(word)
        if typo_type == 'swap' and len(word) > 1:
            i, j = random.sample(range(len(word)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        elif typo_type == 'insert':
            chars.insert(char_idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
        elif typo_type == 'delete':
            del chars[char_idx]
        elif typo_type == 'replace':
            chars[char_idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        words[word_idx] = "".join(chars)
        return words

    def corrupt_subject_verb_agreement(self, words: List[str]) -> List[str]:
        for i, word in enumerate(words):
            if word == "is": words[i] = "are"; return words
            if word == "are": words[i] = "is"; return words
            if word == "was": words[i] = "were"; return words
            if word == "were": words[i] = "was"; return words
            if word == "has": words[i] = "have"; return words
            if word == "have" and i > 0 and words[i - 1].lower() not in ['i', 'we', 'you', 'they']: words[
                i] = "has"; return words
            if word.endswith('s') and len(word) > 2 and word not in self.aux_verbs:
                words[i] = word[:-1]
                return words
        return words

    def delete_word_type(self, words: List[str]) -> List[str]:
        word_type_to_delete = random.choice([self.articles, self.prepositions, self.aux_verbs])
        possible_indices = [i for i, word in enumerate(words) if word in word_type_to_delete]
        if possible_indices:
            idx_to_delete = random.choice(possible_indices)
            del words[idx_to_delete]
        return words

    def delete_punctuation(self, words: List[str]) -> List[str]:
        for i, word in enumerate(words):
            if ',' in word:
                words[i] = word.replace(',', '')
                return words
        if words and words[-1].endswith('.'):
            words[-1] = words[-1][:-1]
        return words

    def add_redundant_word(self, words: List[str]) -> List[str]:
        if not words: return words
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])
        return words

    def add_unnecessary_punctuation(self, words: List[str]) -> List[str]:
        if len(words) > 1:
            idx = random.randint(0, len(words) - 2)
            if not words[idx].endswith(','):
                words[idx] += ','
        return words

    def swap_adjacent_words(self, words: List[str]) -> List[str]:
        if len(words) < 2: return words
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return words

    def apply_random_corruption(self, words: List[str]) -> List[str]:
        method = random.choice(self.corruption_methods)
        return method(words)


# --- Main Generation Script ---

if __name__ == "__main__":
    # --- Configuration ---
    OUTPUT_FILE = "data/gec_dataset_from_c4.tsv"

    # Set this to the desired size of your final dataset.
    # The script will stop once this many samples are generated.
    NUM_SAMPLES = 100000  # Generate 100k samples for a decent start

    # Error application settings
    MIN_ERRORS = 2
    MAX_ERRORS = 4

    # Sentence filtering settings
    MIN_WORDS = 8
    MAX_WORDS = 50

    # --- Initialization ---
    engine = CorruptionEngine()

    print("Loading C4 dataset in streaming mode. This will not download the whole dataset.")
    # We use 'en.noblocklist' which is a cleaner English subset of C4.
    # `streaming=True` is essential to avoid downloading terabytes of data.
    streamed_dataset = load_dataset('c4', 'en.noblocklist', streaming=True, split='train', trust_remote_code=True)

    generated_count = 0

    print(f"Starting dataset generation. Target: {NUM_SAMPLES} samples.")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f, tqdm(total=NUM_SAMPLES) as pbar:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["incorrect_sentence", "correct_sentence"])  # Header

        # Iterate through documents in the C4 stream
        for doc in streamed_dataset:
            if generated_count >= NUM_SAMPLES:
                break  # Stop when we have enough samples

            # Extract text and split into sentences using NLTK for accuracy
            text = doc['text']
            sentences = nltk.sent_tokenize(text)

            for correct_sentence in sentences:
                if generated_count >= NUM_SAMPLES:
                    break

                # --- 1. Filter Sentences ---
                # Remove newlines and excess whitespace
                clean_sentence = correct_sentence.replace('\n', ' ').strip()
                words = clean_sentence.split()

                # Skip sentences that are too short, too long, or contain lists/code snippets.
                if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                    continue
                if any(char in clean_sentence for char in ['{', '}', '<', '>', '=', '*']):
                    continue

                # --- 2. Apply Corruptions ---
                # Pre-process: lowercase and split into words/punctuation.
                words_and_punc = re.findall(r"[\w']+|[.,!?;]", clean_sentence.lower())

                num_errors_to_apply = random.randint(MIN_ERRORS, MAX_ERRORS)
                corrupted_words = words_and_punc.copy()
                applied_errors_count = 0
                attempts = 0  # Safety break

                while applied_errors_count < num_errors_to_apply and attempts < 20:
                    words_before_corruption = corrupted_words.copy()
                    corrupted_words = engine.apply_random_corruption(corrupted_words)

                    if corrupted_words != words_before_corruption:
                        applied_errors_count += 1
                    attempts += 1

                # --- 3. Save the Pair ---
                if applied_errors_count >= MIN_ERRORS:
                    incorrect_sentence = " ".join(corrupted_words).replace(" ,", ",").replace(" .", ".").strip()

                    if incorrect_sentence and incorrect_sentence != clean_sentence.lower():
                        writer.writerow([incorrect_sentence, clean_sentence])
                        generated_count += 1
                        pbar.update(1)

    print(f"\nDataset generation complete. {generated_count} samples saved to {OUTPUT_FILE}")