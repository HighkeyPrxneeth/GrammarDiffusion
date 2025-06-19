import random
import csv
import re
from typing import List, Tuple


class CorruptionEngine:
    """
    A comprehensive engine to programmatically introduce a wide variety of grammatical
    and spelling errors into a sentence.
    """

    def __init__(self):
        # --- Word Lists for Various Error Types ---

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

        # A dictionary of all possible corruption methods.
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

    # --- Replacement Errors ---

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

    # --- Deletion Errors ---

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

    # --- Addition Errors ---

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

    # --- Reordering Errors ---

    def swap_adjacent_words(self, words: List[str]) -> List[str]:
        if len(words) < 2: return words
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return words

    # --- Master Corruption Method ---

    def apply_random_corruption(self, words: List[str]) -> List[str]:
        """Selects and applies one of the defined corruption methods at random."""
        method = random.choice(self.corruption_methods)
        return method(words)


# --- Main Generation Script ---

if __name__ == "__main__":
    # --- Configuration ---
    CORRECT_SENTENCES = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore, and the shells she sells are surely seashells.",
        "I am learning to build a diffusion model from scratch, which is a fascinating process.",
        "The weather is beautiful today in the city; I think I will go for a walk.",
        "He decided to write a book about his grand adventures across the seven seas.",
        "They are planning a trip to the mountains next month if they can get time off work.",
        "What is your favorite type of music to listen to while you are studying?",
        "The cat, which is very fluffy, is sleeping peacefully on the new sofa.",
        "My friend and I are going to the movies tonight to watch the latest blockbuster.",
        "Proper grammar is critically important for clear and effective communication.",
        "Despite the heavy rain, the dedicated team continued to work on the project.",
        "Could you please tell me where the nearest library is located?"
    ]

    OUTPUT_FILE = "data/gec_dataset.tsv"
    NUM_SAMPLES = 5000
    MIN_ERRORS = 2  # The minimum number of errors to apply to each sentence
    MAX_ERRORS = 4  # The maximum number of errors to apply to each sentence

    engine = CorruptionEngine()

    print(f"Generating {NUM_SAMPLES} sentence pairs, each with {MIN_ERRORS}-{MAX_ERRORS} errors...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["incorrect_sentence", "correct_sentence"])  # Header

        generated_count = 0
        while generated_count < NUM_SAMPLES:
            correct_sentence = random.choice(CORRECT_SENTENCES)
            words = re.findall(r"[\w']+|[.,!?;]", correct_sentence.lower())

            # Determine the target number of errors for this specific sentence
            num_errors_to_apply = random.randint(MIN_ERRORS, MAX_ERRORS)

            corrupted_words = words.copy()
            applied_errors_count = 0
            attempts = 0  # Safety break to prevent infinite loops

            # This loop ensures that we successfully apply the target number of distinct errors.
            while applied_errors_count < num_errors_to_apply and attempts < 15:
                # Make a copy to check if the corruption method actually made a change
                words_before_corruption = corrupted_words.copy()

                # Apply a random corruption
                corrupted_words = engine.apply_random_corruption(corrupted_words)

                # Check if the sentence was actually modified. This is key.
                if corrupted_words != words_before_corruption:
                    applied_errors_count += 1

                attempts += 1

            # Only write the sample if we successfully applied enough errors
            if applied_errors_count >= MIN_ERRORS:
                # Post-process: join words back into a sentence, handling punctuation spacing correctly.
                incorrect_sentence = " ".join(corrupted_words).replace(" ,", ",").replace(" .", ".").strip()

                # Final check to ensure we are not writing an empty or unchanged sentence
                if incorrect_sentence and incorrect_sentence != correct_sentence.lower():
                    writer.writerow([incorrect_sentence, correct_sentence])
                    generated_count += 1

                    if generated_count % 500 == 0:
                        print(f"  ... {generated_count}/{NUM_SAMPLES} generated.")

    print(f"\nDataset with {NUM_SAMPLES} complex samples successfully generated at {OUTPUT_FILE}")