from pathlib import Path
import re
import json
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import math


class OneRowKeyboardModel:
    def __init__(self):
        """
        Initialize single-row keyboard model
        Combines dictionary lookup, frequency statistics, and context analysis
        """
        # Keyboard mapping
        self.key_mapping = {
            '2': 'ABC', '3': 'DEF', '4': 'GHI', '5': 'JKL',
            '6': 'MNO', '7': 'PQRS', '8': 'TUV', '9': 'WXYZ'
        }

        # Reverse mapping: letter to number
        self.letter_to_key = {}
        for key, letters in self.key_mapping.items():
            for letter in letters:
                self.letter_to_key[letter] = key

        # Model components
        self.word_dict = defaultdict(list)  # Number sequence -> possible word list
        self.word_freq = defaultdict(int)  # Word frequency
        self.bigram_freq = defaultdict(int)  # Bigram frequency
        self.trigram_freq = defaultdict(int)  # Trigram frequency

        print("Model initialization completed")

    def load_training_data(self, training_file="training_data.txt"):
        """
        Load training data and build dictionary and language model
        """
        print("Loading training data...")

        # BASE_DIR是项目根目录
        BASE_DIR = Path(__file__).resolve().parent.parent
        TRAIN_PATH = BASE_DIR / "datasets" / training_file

        print("Loading file:", TRAIN_PATH)

        with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        words_list = []
        for line in lines:
            line = line.strip()
            if '/' in line:
                numbers, word = line.split('/', 1)

                # Build dictionary
                if word not in self.word_dict[numbers]:
                    self.word_dict[numbers].append(word)

                # Count word frequency
                self.word_freq[word] += 1
                words_list.append(word)

        # Build n-gram model
        self._build_ngram_model(words_list)

        # Sort dictionary by frequency
        for numbers in self.word_dict:
            self.word_dict[numbers].sort(key=lambda w: self.word_freq[w], reverse=True)

        print(f"Training completed:")
        print(f"  - Dictionary size: {len(self.word_dict)}")
        print(f"  - Total vocabulary: {len(self.word_freq)}")
        print(f"  - Bigram count: {len(self.bigram_freq)}")
        print(f"  - Trigram count: {len(self.trigram_freq)}")

    def _build_ngram_model(self, words_list):
        """
        Build n-gram language model
        """
        # Build bigrams
        for i in range(len(words_list) - 1):
            bigram = (words_list[i], words_list[i + 1])
            self.bigram_freq[bigram] += 1

        # Build trigrams
        for i in range(len(words_list) - 2):
            trigram = (words_list[i], words_list[i + 1], words_list[i + 2])
            self.trigram_freq[trigram] += 1

    def get_word_candidates(self, number_sequence: str) -> List[str]:
        """
        Get candidate words based on number sequence
        """
        return self.word_dict.get(number_sequence, [])

    def calculate_word_score(self, word: str, context: List[str] = None) -> float:
        """
        Calculate word score, combining word frequency and context
        """
        # Base frequency score
        freq_score = math.log(self.word_freq[word] + 1)

        # Context score
        context_score = 0
        if context:
            # Check bigram
            if len(context) >= 1:
                bigram = (context[-1], word)
                if bigram in self.bigram_freq:
                    context_score += math.log(self.bigram_freq[bigram] + 1)

            # Check trigram
            if len(context) >= 2:
                trigram = (context[-2], context[-1], word)
                if trigram in self.trigram_freq:
                    context_score += math.log(self.trigram_freq[trigram] + 1) * 1.5

        return freq_score + context_score

    def decode_sequence(self, number_sequences: List[str], use_context=True) -> List[str]:
        """
        Decode number sequences to word sequences
        Uses dynamic programming and context information
        """
        if not number_sequences:
            return []

        n = len(number_sequences)

        # Get candidate words for each position
        candidates = []
        for seq in number_sequences:
            words = self.get_word_candidates(seq)
            if not words:
                # If no candidates found, try to generate possible words
                words = self._generate_possible_words(seq)
            candidates.append(words)

        if not use_context or n == 1:
            # Simple mode: select word with highest frequency
            result = []
            for i, words in enumerate(candidates):
                if words:
                    best_word = max(words, key=lambda w: self.word_freq[w])
                    result.append(best_word)
                else:
                    result.append(f"UNK_{number_sequences[i]}")
            return result

        # Use dynamic programming for context-aware decoding
        return self._dynamic_decode(candidates, number_sequences)

    def _dynamic_decode(self, candidates: List[List[str]], number_sequences: List[str]) -> List[str]:
        """
        Use dynamic programming for decoding
        """
        n = len(candidates)

        # dp[i][j] = (score, path) best score and path when choosing j-th candidate at position i
        dp = [[(-float('inf'), [])] * len(candidates[i]) for i in range(n)]

        # Initialize first position
        for j, word in enumerate(candidates[0]):
            score = self.calculate_word_score(word)
            dp[0][j] = (score, [word])

        # Dynamic programming table filling
        for i in range(1, n):
            for j, curr_word in enumerate(candidates[i]):
                best_score = -float('inf')
                best_path = []

                # Consider all choices from previous position
                for k, prev_word in enumerate(candidates[i - 1]):
                    if dp[i - 1][k][0] == -float('inf'):
                        continue

                    # Calculate current word score
                    prev_path = dp[i - 1][k][1]
                    context = prev_path[-2:] if len(prev_path) >= 2 else prev_path
                    curr_score = self.calculate_word_score(curr_word, context)

                    total_score = dp[i - 1][k][0] + curr_score

                    if total_score > best_score:
                        best_score = total_score
                        best_path = prev_path + [curr_word]

                dp[i][j] = (best_score, best_path)

        # Find best path
        best_score = -float('inf')
        best_result = []

        for j in range(len(candidates[n - 1])):
            if dp[n - 1][j][0] > best_score:
                best_score = dp[n - 1][j][0]
                best_result = dp[n - 1][j][1]

        return best_result if best_result else [f"UNK_{seq}" for seq in number_sequences]

    def _generate_possible_words(self, number_sequence: str) -> List[str]:
        """
        Generate possible words for unknown number sequences
        """
        if not number_sequence:
            return []

        def backtrack(pos: int, current_word: str) -> List[str]:
            if pos == len(number_sequence):
                return [current_word] if current_word else []

            digit = number_sequence[pos]
            if digit not in self.key_mapping:
                return backtrack(pos + 1, current_word)

            results = []
            for letter in self.key_mapping[digit]:
                results.extend(backtrack(pos + 1, current_word + letter.lower()))

            return results

        possible_words = backtrack(0, "")

        # Filter: only keep combinations that might be real words
        # More complex filtering logic can be added here
        return possible_words[:10]  # Limit number of candidates

    def decode_text(self, input_text: str) -> str:
        """
        Decode complete input text
        """
        # Split number sequences
        number_sequences = input_text.strip().split()

        # Decode
        decoded_words = self.decode_sequence(number_sequences)

        return ' '.join(decoded_words)

    def save_model(self, model_file="keyboard_model.json"):
        """
        Save model to file
        """
        model_data = {
            'word_dict': dict(self.word_dict),
            'word_freq': dict(self.word_freq),
            'bigram_freq': {f"{k[0]}|{k[1]}": v for k, v in self.bigram_freq.items()},
            'trigram_freq': {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.trigram_freq.items()}
        }

        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        print(f"Model saved to: {model_file}")

    def load_model(self, model_file="keyboard_model.json"):
        """
        Load model from file
        """
        with open(model_file, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        self.word_dict = defaultdict(list, model_data['word_dict'])
        self.word_freq = defaultdict(int, model_data['word_freq'])

        # Reconstruct n-gram dictionaries
        self.bigram_freq = defaultdict(int)
        for key, value in model_data['bigram_freq'].items():
            words = key.split('|')
            self.bigram_freq[(words[0], words[1])] = value

        self.trigram_freq = defaultdict(int)
        for key, value in model_data['trigram_freq'].items():
            words = key.split('|')
            self.trigram_freq[(words[0], words[1], words[2])] = value

        print(f"Model loaded from {model_file}")


class ModelEvaluator:
    def __init__(self, model: OneRowKeyboardModel):
        self.model = model

    def evaluate_accuracy(self, test_file="test_data.txt") -> Dict[str, float]:
        """
        Evaluate model accuracy
        """
        print("Evaluating model accuracy...")

        BASE_DIR = Path(__file__).resolve().parent.parent
        TEST_PATH = BASE_DIR / "datasets" / test_file

        print("Loading test data from:", TEST_PATH)

        test_cases = []
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse test data
        sections = content.split('---')
        for section in sections:
            lines = section.strip().split('\n')
            if len(lines) >= 2:
                input_line = lines[0].replace('Input: ', '').strip()
                output_line = lines[1].replace('Output: ', '').strip()
                test_cases.append((input_line, output_line))

        # Evaluation metrics
        total_cases = len(test_cases)
        exact_matches = 0
        word_level_correct = 0
        total_words = 0

        results = []

        for input_seq, expected_output in test_cases:
            predicted_output = self.model.decode_text(input_seq)

            # Exact match
            if predicted_output == expected_output:
                exact_matches += 1

            # Word-level accuracy
            expected_words = expected_output.split()
            predicted_words = predicted_output.split()

            for i in range(min(len(expected_words), len(predicted_words))):
                if expected_words[i] == predicted_words[i]:
                    word_level_correct += 1

            total_words += len(expected_words)

            results.append({
                'input': input_seq,
                'expected': expected_output,
                'predicted': predicted_output,
                'exact_match': predicted_output == expected_output
            })

        # Calculate metrics
        sentence_accuracy = exact_matches / total_cases if total_cases > 0 else 0
        word_accuracy = word_level_correct / total_words if total_words > 0 else 0

        evaluation_results = {
            'sentence_accuracy': sentence_accuracy,
            'word_accuracy': word_accuracy,
            'total_cases': total_cases,
            'exact_matches': exact_matches,
            'total_words': total_words,
            'correct_words': word_level_correct,
            'detailed_results': results
        }

        return evaluation_results

    def evaluate_speed(self, test_sequences: List[str], num_runs=10) -> Dict[str, float]:
        """
        Evaluate model speed
        """
        print("Evaluating model speed...")

        times = []
        for _ in range(num_runs):
            start_time = time.time()
            for seq in test_sequences:
                self.model.decode_text(seq)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        sequences_per_second = len(test_sequences) / avg_time

        return {
            'avg_time_per_run': avg_time,
            'sequences_per_second': sequences_per_second,
            'total_sequences': len(test_sequences),
            'num_runs': num_runs
        }


def main():
    """
    Main function: demonstrate model training and evaluation
    """
    # Create model
    model = OneRowKeyboardModel()

    # Load training data
    model.load_training_data("training_data.txt")

    # Save model
    model.save_model()

    # Create evaluator
    evaluator = ModelEvaluator(model)

    # Evaluate accuracy
    accuracy_results = evaluator.evaluate_accuracy("test_data.txt")

    print("\n=== Model Evaluation Results ===")
    print(f"Sentence-level accuracy: {accuracy_results['sentence_accuracy']:.3f}")
    print(f"Word-level accuracy: {accuracy_results['word_accuracy']:.3f}")
    print(f"Test samples: {accuracy_results['total_cases']}")
    print(f"Exact matches: {accuracy_results['exact_matches']}")

    # Show some specific examples
    print("\n=== Prediction Examples ===")
    for i, result in enumerate(accuracy_results['detailed_results'][:5]):
        print(f"\nExample {i + 1}:")
        print(f"  Input: {result['input']}")
        print(f"  Expected: {result['expected']}")
        print(f"  Predicted: {result['predicted']}")
        print(f"  Correct: {'✓' if result['exact_match'] else '✗'}")

    # Evaluate speed
    test_sequences = [result['input'] for result in accuracy_results['detailed_results'][:10]]
    speed_results = evaluator.evaluate_speed(test_sequences)

    print(f"\n=== Performance Evaluation ===")
    print(f"Average processing time: {speed_results['avg_time_per_run']:.4f} seconds")
    print(f"Processing speed: {speed_results['sequences_per_second']:.2f} sequences/second")

    # Interactive testing
    print("\n=== Interactive Testing ===")
    print("Enter number sequences for testing (enter 'quit' to exit):")

    while True:
        user_input = input("\nEnter number sequence: ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            result = model.decode_text(user_input)
            print(f"Decoding result: {result}")


if __name__ == "__main__":
    main()