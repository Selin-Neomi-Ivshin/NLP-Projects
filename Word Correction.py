import re
import random
import math
from collections import *


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,  lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None
        self.error_probabilities = None


    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm



    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables


    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               Float. The float should reflect the (log) probability.
        """
        if not self.lm:
            # If no language model is available return a dummy value
            return float('-inf')
        return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        # If the language model or error tables are not loaded, return the input as-is
        if not self.lm or not self.error_tables:
            return text

        # Normalize the input sentence (e.g., lowercase, strip spaces)
        text = normalize_text(text)

        # Compute error probabilities from confusion matrices and vocabulary statistics
        self.error_probabilities = self.compute_error_probabilities(self.error_tables, self.lm.vocab_counts)

        # --------------------- When the given n is bigger than the given text ---------------->
        if len(text.split()) < self.lm.n:
            # word_probs = self.compute_prior_probs(self.lm.word_count)
            # Tokenize input text into words
            words = text.split()
            # Default: assume the original sentence is the be
            best_sentence = text

            # For each word, assume it may be the incorrect one
            for i, word in enumerate(words):
                # If a better correction was already found, break early (at most one word correction is allowed)
                if best_sentence != text:
                    break
                else:
                    # With probability alpha, keep original
                    best_score = float('-inf')  # Initialize best score to negative infinity when checking new word
                    candidates = self.get_candidates(word)  # Generate candidate corrections for the current word

                    # Evaluate each candidate correction
                    for cand in candidates:
                        #
                        if len(cand) == 1 and cand not in {'a', 'i'}:
                            continue

                        # Replace the i-th word with the candidate
                        new_sentence = words[:i] + [cand] + words[i + 1:]
                        new_text = ' '.join(new_sentence)

                        # Like was seen in the lecture we will do calculation of Ci /N + V
                        lm_score = self.get_smoothed_probability(cand)

                        # Compute the channel model log probability
                        if cand == word:
                            # If the candidate is the same as the original word, use log(alpha)
                            channel_log_prob = math.log(alpha)
                        else:
                            # Otherwise, use log P(observed | candidate) from the error model
                            channel_log_prob = self.get_edit_log_prob(cand, word) + math.log(1 - alpha)

                        # Combine both scores (noisy channel model): total score = log P(sentence) + log P(error)
                        total_score = lm_score + channel_log_prob

                        # Update the best correction if this candidate is better
                        if total_score > best_score:
                            best_score = total_score
                            best_sentence = new_text

            # Return the corrected sentence
            return ''.join(best_sentence)

        # --------------------- When the given text is bigger than the given n ---------------->
        else:
            # Tokenize input text into words
            words = text.split()
            # Default: assume the original sentence is the be
            best_sentence = text

            # For each word, assume it may be the incorrect one
            for i, word in enumerate(words):
                # If a better correction was already found, break early (at most one word correction is allowed)
                print(word)
                if best_sentence != text:
                    break

                else:
                    # With probability alpha, keep original
                    best_score = float('-inf')  # Initialize best score to negative infinity when checking new word
                    candidates = self.get_candidates(word)  # Generate candidate corrections for the current word

                    # Evaluate each candidate correction
                    for cand in candidates:
                        #
                        if len(cand) == 1 and cand not in {'a', 'i'}:
                            continue

                        # Replace the i-th word with the candidate
                        new_sentence = words[:i] + [cand] + words[i + 1:]
                        new_text = ' '.join(new_sentence)

                        # Compute the language model score (log-likelihood of the full sentence)
                        lm_score = self.evaluate_text(new_text)

                        # Compute the channel model log probability
                        if cand == word:
                            # If the candidate is the same as the original word, use log(alpha)
                            channel_log_prob = math.log(alpha)
                        else:
                            # Otherwise, use log P(observed | candidate) from the error model
                            channel_log_prob = self.get_edit_log_prob(cand, word) + math.log(1 - alpha)

                        # Combine both scores (noisy channel model): total score = log P(sentence) + log P(error)
                        total_score = lm_score + channel_log_prob

                        # Update the best correction if this candidate is better
                        if total_score > best_score:
                            best_score = total_score
                            best_sentence = new_text

            # Return the corrected sentence
            return ''.join(best_sentence)

    def get_smoothed_probability(self, cand):
        """
        Calculates the Laplace-smoothed probability of a word using the language model.

        Args:
            word (str): The candidate word to evaluate.

        Returns:
            float: Smoothed probability of the word.
        """
        # Get the raw count of the word from the language model (0 if not found)
        word_count = self.lm.model_dict.get(cand, 0)

        # Get the total number of unique words in the vocabulary
        vocab_size_unique = self.lm.vocab_size

        # Get the total number of tokens (words) in the corpus
        total_tokens = self.lm.corpus_size

        # Apply Laplace smoothing
        smoothed_prob = (word_count + 1) / (total_tokens + vocab_size_unique)

        # Apply smoothing
        return smoothed_prob

    #The functions known, edits1, edits2 and get_candidates were taken from the practice
    #The only change in the get candidates func is that at fist we check if there are candidates in edits1 than edits2 and only then if the word is in the vocabulary
    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.lm.vocab)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def get_candidates(self, word):
        "Generate possible spelling corrections for word."
        if self.known(self.edits1(word)):
            return self.known(self.edits1(word))
        elif self.known(self.edits2(word)):
            return self.known(self.edits2(word))
        else:
            return self.known([word])

    def get_edit_log_prob(self, correct, observed):
        """
        Calculates the log probability that `observed` is a corrupted version of `correct`
        using the noisy channel model and known error probabilities.

        Args:
            correct (str): the candidate correct word
            observed (str): the misspelled word

        Returns:
            float: log-probability of generating observed from correct
        """
        prob = 1.0  # start with probability = 1 (will be multiplied by error probabilities)
        i, j = 0, 0  # indexes for characters in correct and observed

        while i < len(correct) and j < len(observed):
            if correct[i] == observed[j]:
                # Characters match → no error here
                i += 1
                j += 1
            else:
                # Possible substitution (change of a character)
                if i + 1 <= len(correct) and j + 1 <= len(observed) and correct[i + 1:] == observed[j + 1:]:
                    key = correct[i] + observed[j]  # e.g. 'o' → 'e' becomes 'oe'

                    # If not in error table, fallback to frequency or 1
                    if key not in self.error_tables['substitution'] and key in self.lm.vocab_counts:
                        prob = 1 / self.lm.vocab_counts[key]
                    elif key not in self.error_tables['substitution'] and key not in self.lm.vocab_counts:
                        prob = 1
                    else:
                        prob *= self.error_probabilities['substitution'].get(key, 0)
                    i += 1
                    j += 1

                # Possible deletion (a character from correct is missing in observed)
                elif i + 1 <= len(correct) and correct[i + 1:] == observed[j:]:
                    key = correct[i] + correct[i + 1] if i + 1 < len(correct) else correct[i]
                    if key not in self.error_tables['deletion'] and key in self.lm.vocab_counts:
                        prob = 1 / self.lm.vocab_counts[key]
                    elif key not in self.error_tables['deletion'] and key not in self.lm.vocab_counts:
                        prob = 1
                    else:
                        prob *= self.error_probabilities['deletion'].get(key, 0)
                    i += 1

                # Possible insertion (extra character in observed that’s not in correct)
                elif j + 1 <= len(observed) and correct[i:] == observed[j + 1:]:
                    key = observed[j - 1] + observed[j] if j > 0 else '#' + observed[j]
                    if key not in self.error_tables['insertion'] and key in self.lm.vocab_counts:
                        prob = 1 / self.lm.vocab_counts[key]
                    elif key not in self.error_tables['insertion'] and key not in self.lm.vocab_counts:
                        prob = 1
                    else:
                        prob *= self.error_probabilities['insertion'].get(key, 0)
                    j += 1

                # Possible transposition (swapped characters)
                elif (i + 1 < len(correct) and j + 1 < len(observed) and
                      correct[i] == observed[j + 1] and correct[i + 1] == observed[j]):
                    key = correct[i] + correct[i + 1]
                    if key not in self.error_tables['transposition'] and key in self.lm.vocab_counts:
                        prob = 1 / self.lm.vocab_counts[key]
                    elif key not in self.error_tables['transposition'] and key not in self.lm.vocab_counts:
                        prob = 1
                    else:
                        prob *= self.error_probabilities['transposition'].get(key, 1e-5)
                    i += 2
                    j += 2

                # Unrecognized or compound error → apply small fallback probability
                else:
                    prob *= 1e-5
                    i += 1
                    j += 1

        # If there are remaining characters in correct → treat as deletions
        if i < len(correct):
            for k in range(i, len(correct)):
                key = correct[k - 1] + correct[k] if k > 0 else '#' + correct[k]
                prob *= self.error_probabilities['deletion'].get(key, 1e-5)

        # If there are remaining characters in observed → treat as insertions
        elif j < len(observed):
            for k in range(j, len(observed)):
                key = observed[k - 1] + observed[k] if k > 0 else '#' + observed[k]
                prob *= self.error_probabilities['insertion'].get(key, 1e-5)

        # Return log probability (add epsilon to prevent log(0))
        return math.log(prob + 1e-10)


    def compute_error_probabilities(self, error_tables, vocab_counts):
        """
        Computes the probabilities of different character-level errors using observed error counts
        and vocabulary statistics. Applies error-type-specific denominators based on character or bigram frequency.

        Args:
            error_tables (dict): Dictionary of error types to error count mappings.
                Example format:
                {
                    'insertion': {'ga': 5, 'tr': 3, ...},
                    'deletion': {'th': 2, 're': 4, ...},
                    'substitution': {'a>e': 10, ...},
                    ...
                }

            vocab_counts (dict): Dictionary with unigram and bigram frequency counts,
                e.g., from `get_vocab_counts_from_text`. Used to normalize error frequencies into probabilities.

        Returns:
            dict: A nested dictionary of the form:
                {
                    'insertion': { 'ga': P(inserting 'a' after 'g'), ... },
                    'deletion': { 'th': P(deleting 'h' after 't'), ... },
                    'substitution': { 'a>e': P(substituting 'a' with 'e'), ... },
                    ...
                }
        """

        # Final output: dictionary to hold all computed probabilities per error type
        error_probs = {}

        # Loop over each error type and its corresponding error count dictionary
        for err_type, err_counts in error_tables.items():
            probs = {}  # Dictionary to hold probabilities for this specific error type
            for key, count in err_counts.items():
                # Determine the denominator based on error type:
                # For insertion, deletion, and transposition: use bigram count (e.g., 'ga')
                # For substitution: use unigram count of the correct character (e.g., 'a' in 'a>e')
                if err_type in ('insertion', 'deletion', 'transposition'):
                    denom = vocab_counts.get(key, 0) # Bigram frequency
                elif err_type == 'substitution':
                    denom = vocab_counts.get(key[1], 0) # Frequency of the correct character being replaced
                else:
                    denom = 1 # Fallback (should rarely happen unless error type is unrecognized)

                # Compute smoothed probability:
                # P(error | context) = count(error) / count(context)
                # If context never occurred (denom = 0), fall back to a probability of 1.0 (very unlikely)
                probs[key] = count / denom if denom > 0 else 1.0

                if denom != 0: # In case our corpus is small and the division gives probability bigger than 1
                    if (count / denom) > 1:
                        probs[key] = 1e-3
            # Store computed probabilities for this error type
            error_probs[err_type] = probs
        return error_probs


    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict = None #a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            #NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.
            self.chars = chars
            self.context_dict = {}  # Dictionary for (n-1)-gram counts (contexts)
            self.vocab = None      # Set of all tokens in the vocabulary
            self.vocab_size = 0     # Size of the vocabulary
            self.vocab_counts = None # Counts of unigrams and bigrams (used for error probabilities)
            self.corpus_size = 0 #Holds the number of all the words in the corpus


        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            # Initialize the model dictionary
            self.model_dict = {}
            self.context_dict = {}
            self.vocab = set()

            # Normalize the text
            text = normalize_text(text)
            self.corpus_size = len(text.split())
            # Normalize and tokenize the corpus (only alphabetic words)


            # Get unigram and bigram counts (used in other parts of the system)
            self.vocab_counts = self.get_vocab_counts_from_text(text)

            # Process text based on character or word model
            if self.chars:
                # For character model, each character is a token
                tokens = list(text)
            else:
                # For word model, split by spaces
                tokens = text.split()

            # Add start and end markers
            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            # Build vocabulary, excluding special tokens
            self.vocab= set(t for t in padded_tokens if t not in ("<s>", "</s>"))

            # In character mode, remove space from vocabulary
            if self.chars:
                if ' ' in self.vocab:
                    self.vocab.remove(' ')
            # Save vocabulary size
            self.vocab_size = len(self.vocab)

            # Create the n-gram and (n-1)-gram counts
            for i in range(len(padded_tokens) - self.n + 1):
                # Get current n-gram tuple
                ngram = tuple(padded_tokens[i:i + self.n])

                # Update count of this n-gram in model_dict
                self.model_dict[ngram] = self.model_dict.get(ngram, 0) + 1

                # Get context (first n-1 elements of the n-gram)
                context = ngram[:-1]

                # Update context count
                self.context_dict[context] = self.context_dict.get(context, 0) + 1

        def get_vocab_counts_from_text(self, text):
            """
            Computes unigram and bigram frequency counts from the text. This function will
            help in the calculation of the error probabilities dictionary.

            Returns:
                unigram_counts: dict mapping character to frequency
                bigram_counts: dict mapping character pairs (as string) to frequency
            """
            # Initialize frequency dictionaries for unigrams and bigrams
            unigram_counts = defaultdict(int)
            bigram_counts = defaultdict(int)

            # Process each word in the text
            for word in text.split():
                # Add a start-of-word marker '#' to help model insertions at the beginning
                padded_word = "#" + word
                # Iterate over characters in the padded word
                for i in range(len(padded_word)):
                    # Count the unigram (single character)
                    unigram_counts[padded_word[i]] += 1

                    # If there is a next character, count the bigram (pair of characters)
                    if i < len(padded_word) - 1:
                        bigram = padded_word[i] + padded_word[i + 1]
                        bigram_counts[bigram] += 1

            # Convert defaultdicts to regular dicts for return
            unigram_counts = dict(unigram_counts)
            bigram_counts = dict(bigram_counts)
            # Merge the two dictionaries and return the combined result
            return {**unigram_counts, **bigram_counts}

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            # --- Handle the context initialization ---
            if context is None:
                # No context provided: sample a random one from the context_dict
                contexts = list(self.context_dict.keys())

                # If there are no contexts, use default start tokens
                if not contexts:
                    current_context = tuple(['<s>'] * (self.n - 1))
                else:
                    # Sample a context based on frequency (weighted random choice)
                    weights = [self.context_dict[c] for c in contexts]
                    current_context = random.choices(contexts, weights=weights)[0]
                tokens = list(current_context)
            else:
                # Context was provided
                tokens = list(context) if self.chars else context.split()

                # If the context is already long enough, truncate and return early
                if len(tokens) >= n:
                    return ''.join(tokens[:n]) if self.chars else ' '.join(tokens[:n])

                # If too short, pad the context with <s> tokens
                if len(tokens) < self.n - 1:
                    tokens = ['<s>'] * (self.n - 1 - len(tokens)) + tokens

                # Keep only the last (n-1) tokens as current context
                current_context = tuple(tokens[-(self.n - 1):])

            # --- Character-level model generation ---
            if self.chars:
                # Result characters (excluding special tokens like <s>)
                result_chars = [t for t in tokens if t not in ('<s>', '</s>')]

                # Generate up to n characters
                while len(result_chars) < n:
                    # Collect candidate next characters given the current context
                    candidates = {
                        ngram[-1]: count
                        for ngram, count in self.model_dict.items()
                        if ngram[:-1] == current_context
                    }
                    # If no candidates found, stop generation
                    if not candidates:
                        break

                    # Choose next character using weighted random sampling
                    next_token = random.choices(
                        list(candidates.keys()),
                        weights=list(candidates.values())
                    )[0]

                    # Stop if we reached an end token
                    if next_token == '</s>':
                        break
                    # Append the next character and update the current context
                    tokens.append(next_token)
                    if next_token != '<s>': # Don't include start tokens in output
                        result_chars.append(next_token)

                    current_context = tuple(tokens[-(self.n - 1):])

                return ''.join(result_chars[:n])

            # --- Word-level model generation ---
            else:
                result_tokens = [t for t in tokens if t not in ('<s>', '</s>')]

                # Generate up to n words
                while len(result_tokens) < n:
                    # Collect candidate next words given the current context
                    candidates = {
                        ngram[-1]: count
                        for ngram, count in self.model_dict.items()
                        if ngram[:-1] == current_context
                    }
                    # Stop if there are no candidates for this context
                    if not candidates:
                        break

                    # Sample next word
                    next_token = random.choices(
                        list(candidates.keys()),
                        weights=list(candidates.values())
                    )[0]

                    # End token encountered — stop generation
                    if next_token == '</s>':
                        break

                    # Add token and update current context
                    tokens.append(next_token)
                    if next_token != '<s>': # Skip start markers
                        result_tokens.append(next_token)

                    current_context = tuple(tokens[-(self.n - 1):])

                return ' '.join(result_tokens[:n])

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            # Return 0 if text is None
            if text is None:
                return 0

            # Tokenize input based on model type (character vs word level)
            if self.chars:
                tokens = list(text)
            else:
                tokens = text.split()

            # Add n-1 start tokens and an end token for boundary context
            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            # Calculate log probability

            prob = 1 # Start with a neutral probability (will be multiplied)

            # Iterate over all possible n-grams in the text
            for i in range(len(padded_tokens) - self.n + 1):
                # Create the n-gram as a tuple
                ngram = tuple(padded_tokens[i:i + self.n])
                # Get the context (n-1 gram)
                context = ngram[:-1]

                #if the ngram and the context are in the dictionaries we won't do laplace smoothing
                if ngram in self.model_dict and context in self.context_dict:
                    ngram_count = self.model_dict.get(ngram)
                    context_count = self.context_dict.get(context)
                    prob *= ngram_count / context_count # P(w|context) = (count(context, w)) / (count(context))
                else: #if the ngram and the context aren't in the dictionaries we will do laplace smoothing
                    prob *= self.smooth(" ".join(ngram))

            # If the n-gram and its context were seen during training
            return math.log(prob, 10)

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            # Convert string to tuple of tokens (based on model type)
            if isinstance(ngram, str):
                if self.chars:
                    ngram = tuple(ngram)
                else:
                    ngram = tuple(ngram.split())

            # Extract count of the full ngram and its context
            ngram_count = self.model_dict.get(ngram, 0)
            context = ngram[:-1]
            context_count = self.context_dict.get(context, 0)

            # Apply Laplace smoothing:
            # P(w | context) = (count(ngram) + 1) / (count(context) + V)
            # where V = vocabulary size
            smoothed_prob = (ngram_count + 1) / (context_count + self.vocab_size)

            return smoothed_prob

def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    # Step 1: Remove HTML tags manually using regex
    text = re.sub(r'<[^>]+>', '', text)

    # Step 2: Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Step 3: Remove punctuation using regex (basic coverage)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)

    # Step 4: Remove digits and non-alphabetic characters (replace with space)
    text = re.sub(r'\d+', ' ', text)  # Replace digits with space
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove any remaining non-alphabetic chars

    # Step 5: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Step 6: Replace <s> tags with spaces
    text = text.replace('<s>', ' ')

    # Convert to lowercase and strip leading/trailing spaces
    text = text.lower().strip()

    return text

def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Selin Neomi Ivshin', 'id': '322769175', 'email': 'ivshins@post.bgu.ac.il'}
