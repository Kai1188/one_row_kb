import re
import os
import json
import time
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path


class LargeDatasetProcessor:
    def __init__(self):
        """
        Large scale arithmetic machine
        Supported by many sources: local literature, online language libraries, basic encyclopedias, etc.
        """
        self.key_mapping = {
            '2': 'ABC', '3': 'DEF', '4': 'GHI', '5': 'JKL',
            '6': 'MNO', '7': 'PQRS', '8': 'TUV', '9': 'WXYZ'
        }

        self.letter_to_key = {}
        for key, letters in self.key_mapping.items():
            for letter in letters:
                self.letter_to_key[letter] = key

    def download_sample_datasets(self):
        """
        Download sample datasets
        """
        datasets = {
            'gutenberg_alice': 'https://www.gutenberg.org/files/11/11-0.txt',
            'gutenberg_shakespeare': 'https://www.gutenberg.org/files/1513/1513-0.txt',
            'news_sample': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
        }

        os.makedirs('../datasets', exist_ok=True)

        for name, url in datasets.items():
            file_path = f'datasets/{name}.txt'
            if not os.path.exists(file_path):
                try:
                    print(f"Downloading {name}...")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)

                    print(f"✓ {name} download completed")
                except Exception as e:
                    print(f"✗ {name} download failed: {e}")

    def create_large_corpus_from_files(self, input_dir="datasets", output_file="large_corpus.txt"):
        """
        Create large corpus from multiple files
        """
        print("Creating large corpus...")

        all_text = []
        total_chars = 0

        # Process all text files in the directory
        for file_path in Path(input_dir).glob('*.txt'):
            print(f"Processing file: {file_path.name}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Clean text
                cleaned_content = self.clean_text(content)
                all_text.append(cleaned_content)
                total_chars += len(cleaned_content)

                print(f"  - File size: {len(content):,} characters")
                print(f"  - After cleaning: {len(cleaned_content):,} characters")

            except Exception as e:
                print(f"  - Error: {e}")

        # Merge all text
        combined_text = '\n'.join(all_text)

        # Save combined corpus
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        print(f"\nCorpus creation completed:")
        print(f"  - Total characters: {len(combined_text):,}")
        print(f"  - Saved to: {output_file}")

        return output_file

    def clean_text(self, text: str) -> str:
        """
        Clean text data
        """
        # Remove Project Gutenberg headers and footers
        text = re.sub(r'\*\*\* START OF.*?\*\*\*', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\* END OF.*?\*\*\*', '', text, flags=re.DOTALL)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-English characters (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?;:\'"()-]', '', text)

        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n', '\n', text)

        return text.strip()

    def create_training_data_batch(self, corpus_file, batch_size=10000, output_prefix="training_batch"):
        """
        Create training data in batches (for processing large corpus)
        """
        print(f"Processing corpus in batches: {corpus_file}")

        with open(corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into sentences
        sentences = self.split_into_sentences(content)
        print(f"Total sentences: {len(sentences):,}")

        # Process in batches
        batch_num = 0
        current_batch = []
        all_training_pairs = []

        for i, sentence in enumerate(sentences):
            words = self.extract_words(sentence)

            # Create training pairs for each word
            for word in words:
                if len(word) > 1:  # Filter single-character words
                    numbers = self.word_to_numbers(word)
                    if numbers:
                        training_pair = f"{numbers}/{word}"
                        current_batch.append(training_pair)
                        all_training_pairs.append(training_pair)

            # Save batch
            if len(current_batch) >= batch_size:
                batch_file = f"{output_prefix}_{batch_num}.txt"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_batch))

                print(f"Saved batch {batch_num}: {len(current_batch)} training pairs")

                current_batch = []
                batch_num += 1

        # Save the last batch
        if current_batch:
            batch_file = f"{output_prefix}_{batch_num}.txt"
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_batch))
            print(f"Saved final batch {batch_num}: {len(current_batch)} training pairs")

        # Save complete training data
        full_training_file = f"{output_prefix}_full.txt"
        with open(full_training_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_training_pairs))

        print(f"\nTraining data creation completed:")
        print(f"  - Total training pairs: {len(all_training_pairs):,}")
        print(f"  - Number of batch files: {batch_num + 1}")
        print(f"  - Full training file: {full_training_file}")

        return full_training_file, batch_num + 1

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    def extract_words(self, sentence: str) -> List[str]:
        """
        Extract words from sentence
        """
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        return [word for word in words if len(word) > 1]

    def word_to_numbers(self, word: str) -> str:
        """
        Convert word to number sequence
        """
        numbers = []
        for char in word.upper():
            if char in self.letter_to_key:
                numbers.append(self.letter_to_key[char])
            else:
                return ""  # Contains unmappable characters
        return ''.join(numbers)

    def create_challenging_test_set(self, corpus_file, num_tests=1000, output_file="challenging_test.txt"):
        """
        Create more challenging test set
        """
        print(f"Creating challenging test set...")

        with open(corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()

        sentences = self.split_into_sentences(content)

        # Select sentences of different lengths
        test_cases = []
        lengths = [3, 5, 7, 10, 15]  # Different sentence lengths

        for target_length in lengths:
            count = num_tests // len(lengths)

            suitable_sentences = []
            for sentence in sentences:
                words = self.extract_words(sentence)
                if len(words) >= target_length:
                    suitable_sentences.append(words[:target_length])

            if suitable_sentences:
                selected = random.sample(suitable_sentences, min(count, len(suitable_sentences)))

                for words in selected:
                    # Create number sequences
                    number_sequences = []
                    valid_words = []

                    for word in words:
                        numbers = self.word_to_numbers(word)
                        if numbers:
                            number_sequences.append(numbers)
                            valid_words.append(word)

                    if len(number_sequences) >= 3:
                        test_input = ' '.join(number_sequences)
                        test_output = ' '.join(valid_words)
                        test_cases.append((test_input, test_output))

        # Shuffle test set
        random.shuffle(test_cases)

        # Save test set
        with open(output_file, 'w', encoding='utf-8') as f:
            for test_input, test_output in test_cases:
                f.write(f"Input: {test_input}\n")
                f.write(f"Output: {test_output}\n")
                f.write("---\n")

        print(f"Test set creation completed:")
        print(f"  - Number of test cases: {len(test_cases)}")
        print(f"  - Saved to: {output_file}")

        return test_cases

    def analyze_dataset_complexity(self, training_file):
        """
        Analyze dataset complexity
        """
        print("Analyzing dataset complexity...")

        word_freq = defaultdict(int)
        number_to_words = defaultdict(set)

        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '/' in line:
                    numbers, word = line.split('/', 1)
                    word_freq[word] += 1
                    number_to_words[numbers].add(word)

        # Calculate complexity metrics
        total_words = len(word_freq)
        total_number_sequences = len(number_to_words)

        # Ambiguity analysis
        ambiguous_sequences = {k: v for k, v in number_to_words.items() if len(v) > 1}
        ambiguity_levels = defaultdict(int)

        for numbers, words in ambiguous_sequences.items():
            level = len(words)
            ambiguity_levels[level] += 1

        print(f"\n=== Dataset Complexity Analysis ===")
        print(f"Total vocabulary: {total_words:,}")
        print(f"Unique number sequences: {total_number_sequences:,}")
        print(f"Ambiguous sequences: {len(ambiguous_sequences):,}")
        print(f"Average ambiguity: {sum(len(v) for v in ambiguous_sequences.values()) / len(ambiguous_sequences):.2f}")

        print(f"\nAmbiguity distribution:")
        for level in sorted(ambiguity_levels.keys()):
            print(f"  {level} candidate words: {ambiguity_levels[level]} sequences")

        # Show most complex sequences
        most_ambiguous = sorted(ambiguous_sequences.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\nMost complex number sequences:")
        for numbers, words in most_ambiguous[:10]:
            words_list = list(words)[:10]  # Limit display count
            print(f"  {numbers}: {words_list}")

        return {
            'total_words': total_words,
            'total_sequences': total_number_sequences,
            'ambiguous_sequences': len(ambiguous_sequences),
            'avg_ambiguity': sum(len(v) for v in ambiguous_sequences.values()) / len(
                ambiguous_sequences) if ambiguous_sequences else 0,
            'ambiguity_distribution': dict(ambiguity_levels)
        }


def main():
    """
    Main function: Process large datasets
    """
    processor = LargeDatasetProcessor()

    print("=== Large Dataset Processor ===")
    print("1. Download sample datasets")
    print("2. Create large corpus")
    print("3. Generate training data")
    print("4. Create challenging test set")
    print("5. Analyze dataset complexity")

    # Step 1: Download datasets
    processor.download_sample_datasets()

    # Step 2: Create corpus
    corpus_file = processor.create_large_corpus_from_files()

    # Step 3: Generate training data
    training_file, num_batches = processor.create_training_data_batch(corpus_file)

    # Step 4: Create test set
    test_cases = processor.create_challengin_test_set(corpus_file, num_tests=500)

    # Step 5: Analyze complexity
    complexity_stats = processor.analyze_dataset_complexity(training_file)

    print("\n=== Processing Complete ===")
    print(f"Corpus file: {corpus_file}")
    print(f"Training file: {training_file}")
    print(f"Test file: challenging_test.txt")
    print(f"Dataset size: {complexity_stats['total_words']:,} words")
    print(f"Training recommendation: Retrain the model using {training_file}")


if __name__ == "__main__":
    main()