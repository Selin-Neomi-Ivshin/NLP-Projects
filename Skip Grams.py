"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""

# you can use these packages (uncomment as needed)
import pickle
import pandas as pd
import numpy as np
import os, time, re, sys, random, math, collections, nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Selin Neomi Ivshin', 'id': '322769175', 'email': 'ivshins@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []

    # Try to open the file using UTF-8 encoding
    try:
        with open(fn, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try Windows-1252 encoding
        try:
            with open(fn, 'r', encoding='windows-1252') as file:
                text = file.read()
        except UnicodeDecodeError:
            # If that also fails, fall back to Latin-1 encoding
            with open(fn, 'r', encoding='latin-1') as file:
                text = file.read()


    # Replace problematic/smart punctuation characters from Windows-encoded text
    text = text.replace('\x91', "'").replace('\x92', "'")   # smart apostrophes → regular apostrophe
    text = text.replace('\x93', '"').replace('\x94', '"')   # smart quotes → regular quote
    text = text.replace('\x97', ' ')  # em dash → space
    text = text.replace('\x96', '-').replace('\x97', ' ')  # dashes
    text = text.replace('\x85', '...')  # ellipsis
    text = text.replace('\xa0', ' ')  # non-breaking space
    text = text.replace('\x95', '-')  # bullet
    text = text.replace('\x86', '').replace('\x87', '')  # cross symbols

    # Split the cleaned text into sentences using NLTK's sentence tokenizer
    sub_sentences = sent_tokenize(text)

    for sentence in sub_sentences:
        # Normalize remaining smart punctuation inside each sentence
        sentence = re.sub(r'[“”]', '"', sentence)  # smart quotes to regular
        sentence = re.sub(r'[—–−]', ' ', sentence)  # dashes to space
        sentence = re.sub(r'[…]', '...', sentence)  # ellipsis
        sentence = re.sub(r'\n+', ' ', sentence)  # normalize line breaks

        # Convert to lowercase and remove punctuation and digits
        clean_sentence = re.sub(r'[?!.:;"#$%^&*\)\(\-+=\d]', ' ', sentence.lower())
        # Remove extra spaces (multiple spaces → one), trim leading/trailing
        clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
        # Tokenize sentence into words and keep only non-empty tokens
        tokens = [token for token in clean_sentence.split() if token]
        # Only add non-empty token lists (valid sentences)
        if tokens:
            sentences.append(tokens)

    return sentences

def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    # Open the file in binary read mode
    with open(fn, 'rb') as f:
        # Load the pickled model from the file
        sg_model = pickle.load(f)
    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        # Tips:
        # 1. It is recommended to create a word:count dictionary
        # Count word frequencies across all sentences
        self.word_counts = collections.Counter()
        for sentence in sentences:
            self.word_counts.update(sentence)

        # Filter out infrequent words (only keep words with enough occurrences)
        self.vocab = {word: count for word, count in self.word_counts.items()
                      if count >= self.word_count_threshold}

        # 2. It is recommended to create a word-index map
        # Create word-to-index and index-to-word mappings
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Store the size of the vocabulary
        self.vocab_size = len(self.vocab)

        # Prepare the probability distribution for negative sampling
        self.prepare_negative_sampling_distribution()

        # Initialize embeddings to None (will be created during training)
        self.T = None  # Target word embeddings
        self.C = None  # Context word embeddings
        self.V = None  # Combined embeddings

    def prepare_negative_sampling_distribution(self):
        """
        Prepares a probability distribution over the vocabulary
        for negative sampling, based on word frequencies.

        According to the original word2vec paper, using raw frequencies for sampling
        makes common words appear too often as negative samples, which is not helpful.
        Instead, the frequencies are raised to the power of 3/4 to smooth them:
        - Frequent words are down-weighted
        - Rare words are up-weighted
        """
        # Calculate word frequencies raised to power of 3/4 as per original word2vec paper
        word_freqs = np.array([self.word_counts[word] for word in self.vocab])
        word_freqs = word_freqs ** 0.75

        # Normalize to get probability distribution
        self.neg_sample_probs = word_freqs / np.sum(word_freqs)
        # These probabilities will be used later with np.random.choice
        # to sample negative context words efficiently

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default

        # Check that both words exist in the vocabulary
        if w1 not in self.word2idx or w2 not in self.word2idx:
            return sim

        # Retrieve indices of the words in the embedding matrix
        idx1 = self.word2idx[w1]
        idx2 = self.word2idx[w2]

        # Choose the appropriate embedding vectors
        if self.V is not None:
            # Use combined embeddings if available
            vec1 = self.V[idx1]
            vec2 = self.V[idx2]
        elif self.T is not None:
            # Otherwise fall back to target embeddings
            vec1 = self.T[:, idx1]
            vec2 = self.T[:, idx2]
        else:
            # Otherwise fall back to target embeddings
            return sim

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)  # Calculate dot product between the two vectors
        # Calculate L2 norms (magnitudes) of each vector
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Prevent division by zero in case of zero vectors
        if norm1 == 0 or norm2 == 0:
            return sim

        # Compute cosine similarity and normalize it to [0, 1]
        # Standard cosine similarity is in [-1, 1]; we map it to [0, 1]
        sim = float((dot_product / (norm1 * norm2) + 1) / 2)
        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

        # if the given word isn't in the model's dictionary return empty list
        if w not in self.word2idx:
            return []

        # Create a list of (word, similarity score) for all other words in the vocabulary
        similarities = [
            (word, self.compute_similarity(w, word))
            for word in self.word2idx
            if word != w  # Exclude the input word itself
        ]

        # Sort the list of word-similarity pairs by similarity in descending order
        sorted_similar = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Extract only the top-n words (ignore the similarity scores)
        closest_words = [word for word, _ in sorted_similar[:n]]
        return closest_words

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None, flag = False):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        # --------------------------------------- Initialize Embedding Matrices --------------------------------->
        vocab_size = self.vocab_size
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        # ------------------------------------- Data Preparation ----------------------------------------------->
        # A list which will include [(target_index, positive_index, [] - List of corresponding negative samples indexes)]
        data_samples = []
        for sentence in self.sentences:
            # Keep only words that appear in the vocabulary
            filtered_sentence = [word for word in sentence if word in self.word2idx]

            # Skip short or empty sentences
            if len(filtered_sentence) < 2:
                continue

            # For each word, collect context window examples
            for center_pos, center_word in enumerate(filtered_sentence):
                center_idx = self.word2idx[center_word]

                # Determine boundaries of context window
                start = max(0, center_pos - self.context)
                end = min(len(filtered_sentence), center_pos + self.context + 1)

                # Get context word indices (excluding the center word itself)
                context_indices = [self.word2idx[filtered_sentence[pos]]
                                   for pos in range(start, end)
                                   if pos != center_pos]

                # Skip if no context words
                if not context_indices:
                    continue

                # Convert context words to indices
                # For each context word, perform SGD update
                for context_idx in context_indices:
                    # List to hold the negative samples indexes
                    negative_sample = []

                    # Negative sampling - For each positive word index we will extract self.neg_samples negative indexes
                    while len(negative_sample) < self.neg_samples:
                        # We sample the negative indexes by using the probabilities that we have already calculated
                        neg_idx = np.random.choice(vocab_size, p=self.neg_sample_probs)
                        # Skip if we sampled the positive context by chance
                        if neg_idx == context_idx or neg_idx == center_idx:
                            continue
                        # Append the negative index
                        negative_sample.append(neg_idx)
                    # Append the indexes to the list
                    data_samples.append((center_idx, context_idx, negative_sample))

        # --------------------------------------------- Start Training ---------------------------------->

        best_loss = float('inf')  # Track best loss for early stopping
        epochs_no_improve = 0  # Count of epochs with no improvement

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            num_examples = 0

            # Shuffle training examples for better convergence
            random.shuffle(data_samples)

            for center_idx, context_idx, neg_indices in data_samples:
                # ---- POSITIVE EXAMPLE ---->
                center_vec = T[:, center_idx]  # Vector for target (center) word
                context_vec = C[context_idx]  # Vector for context word

                # Dot product and sigmoid: similarity score
                dot_product = np.dot(context_vec, center_vec)
                sigmoid_val = sigmoid(dot_product)
                # Binary cross-entropy loss for the positive pair
                if sigmoid_val > 0:
                    pos_loss = -np.log(sigmoid_val)
                else:
                    pos_loss = -np.log(1e-8)

                # Gradient for positive example
                grad_context = (sigmoid_val - 1) * center_vec
                grad_center = (sigmoid_val - 1) * context_vec

                # ---- NEGATIVE EXAMPLES ---->
                neg_loss = 0
                # Go over the negative samples indexes list for the corresponding positive sample
                for neg_idx in neg_indices:
                    neg_vec = C[neg_idx]  # Get the negative context vector

                    # Score the negative pair (we negate the dot product)
                    neg_dot = np.dot(neg_vec, center_vec)
                    neg_sigmoid = sigmoid(-neg_dot)  # (1 - sigmoid(x)) = sigmoid(-x)
                    # Cross-entropy loss for this negative sample
                    if neg_sigmoid > 0:
                        neg_loss -= np.log(neg_sigmoid)
                    else:
                        neg_loss -= np.log(1e-8)

                    # Gradient for negative example
                    neg_grad_context = (1 - neg_sigmoid) * center_vec
                    neg_grad_center = (1 - neg_sigmoid) * neg_vec

                    # Update negative context vector
                    C[neg_idx, :] -= step_size * neg_grad_context

                    # Accumulate gradient for center word
                    grad_center += neg_grad_center

                # Update vectors
                C[context_idx, :] -= step_size * grad_context
                T[:, center_idx] -= step_size * grad_center

                # Accumulate loss
                example_loss = pos_loss + neg_loss
                epoch_loss += example_loss
                num_examples += 1

            # Calculate average loss for epoch
            avg_epoch_loss = epoch_loss / max(1, num_examples)
            epoch_time = time.time() - start_time

            if flag:
                print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s, avg loss: {avg_epoch_loss:.6f}")

            # --------------------------------- Early Stopping Check -------------------------------->
            # If the current epoch's average loss is lower than the best seen so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss  # Update the best loss
                epochs_no_improve = 0  # Reset the no-improvement counter

                # Save a copy of the best model so far
                if model_path:
                    temp_path = f"{model_path}_epoch{epoch + 1}"
                    self.T = T
                    self.C = C
                    with open(temp_path, 'wb') as f:
                        pickle.dump(self, f)
                    if flag:
                        print(f"Saved model to {temp_path}")
            else:
                epochs_no_improve += 1  # No improvement in loss → increment counter

                # If we reached the early stopping threshold (no improvement for X epochs)
                if epochs_no_improve >= early_stopping:
                    break  # Exit the training loop early

        # --------------------------------- Save Final Model ------------------------------->
        # Save the final version of the model (even if not the best one)
        if model_path:
            self.T = T  # Assign the final learned target embeddings to the class
            self.C = C  # Assign the final learned context embeddings to the class
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)  # Save the complete model to the given path
            if flag:
                print(f"Saved final model to {model_path}")

        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        # Select combination strategy
        if combo == 0:
            # Use only the T embeddings (transpose to shape vocab_size x d)
            V = T.T
        elif combo == 1:
            # Use only the context (C) embeddings
            V = C
        elif combo == 2:
            # Take the average of T and C embeddings (pointwise)
            V = (T.T + C) / 2
        elif combo == 3:
            # Take the sum of T and C embeddings (pointwise)
            V = T.T + C
        elif combo == 4:
            # Concatenate T and C to form a single longer vector (shape: vocab_size x 2d)
            V = np.concatenate((T.T, C), axis=1)
        else:
            # Invalid combo strategy → raise an error
            raise ValueError(f"Invalid combo value: {combo}. Expected 0-4.")

        # Update
        self.T = T  # Store original T
        self.C = C  # Store original C
        self.V = V  # Store combined vector matrix

        # Save only V
        if model_path:
            with open(model_path, 'wb') as f:
                pickle.dump(V, f)
        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        # Return empty string if any word is out-of-vocabulary
        if w1 not in self.word2idx or w2 not in self.word2idx or w3 not in self.word2idx:
            return ""

        # Get word indices
        idx1 = self.word2idx[w1]
        idx2 = self.word2idx[w2]
        idx3 = self.word2idx[w3]

        # Get word vectors based on which embeddings are available
        if self.V is not None:
            vec1 = self.V[idx1]
            vec2 = self.V[idx2]
            vec3 = self.V[idx3]
        elif self.T is not None:
            # Otherwise use target embedding matrix T
            vec1 = self.T[:, idx1]
            vec2 = self.T[:, idx2]
            vec3 = self.T[:, idx3]
        else:
            # No vectors available to perform the analogy
            return None

        # Calculate analogy vector: vec1 - vec2 + vec3
        # Example: vec("king") - vec("man") + vec("woman") ≈ vec("queen")
        analogy_vec = vec1 - vec2 + vec3

        # Find closest word to analogy vector (excluding w1, w2, w3)
        max_sim = -1  # Highest cosine similarity found
        w = None  # Best matching word

        for idx in range(self.vocab_size):
            # Skip the input words
            if idx in [idx1, idx2, idx3]:
                continue

            # Get vector for current word
            if self.V is not None:
                curr_vec = self.V[idx]
            else:
                curr_vec = self.T[:, idx]

            # Compute cosine similarity between analogy vector and current word vector
            dot_product = np.dot(analogy_vec, curr_vec)
            norm1 = np.linalg.norm(analogy_vec)
            norm2 = np.linalg.norm(curr_vec)

            # Avoid division by zero
            if norm1 > 0 and norm2 > 0:
                sim = dot_product / (norm1 * norm2)
                # Update best match if similarity is higher than current max
                if sim > max_sim:
                    max_sim = sim
                    w = self.idx2word[idx]
            else:
                sim = 0
                # Update best match if similarity is higher than current max
                if sim > max_sim:
                    max_sim = sim
                    w = self.idx2word[idx]
        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """
        # Return empty string if any word is out-of-vocabulary
        if w1 not in self.word2idx or w2 not in self.word2idx or w3 not in self.word2idx:
            return False

        # Get word indices
        idx1 = self.word2idx[w1]
        idx2 = self.word2idx[w2]
        idx3 = self.word2idx[w3]

        # Get word vectors based on which embeddings are available
        if self.V is not None:
            vec1 = self.V[idx1]
            vec2 = self.V[idx2]
            vec3 = self.V[idx3]
        elif self.T is not None:
            # Otherwise use target embedding matrix T
            vec1 = self.T[:, idx1]
            vec2 = self.T[:, idx2]
            vec3 = self.T[:, idx3]
        else:
            # No vectors available to perform the analogy
            return False

        # Calculate analogy vector: vec1 - vec2 + vec3
        # Example: vec("king") - vec("man") + vec("woman") ≈ vec("queen")
        analogy_vec = vec1 - vec2 + vec3

        top_n_similarities = []
        for idx in range(self.vocab_size):
            # Skip the input words
            if idx in [idx1, idx2, idx3]:
                continue

            # Get vector for current word
            curr_vec = self.V[idx] if self.V is not None else self.T[:, idx]

            # Compute cosine similarity between analogy vector and current word vector
            dot_product = np.dot(analogy_vec, curr_vec)
            norm1 = np.linalg.norm(analogy_vec)
            norm2 = np.linalg.norm(curr_vec)

            # Avoid division by zero
            sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
            top_n_similarities.append((sim, self.idx2word[idx]))

        # Sort similarities and get top-n
        sorted_similar = sorted(top_n_similarities, key=lambda x: x[0], reverse=True)
        closest_words = [word for _, word in sorted_similar[:n]]

        return w4 in closest_words


