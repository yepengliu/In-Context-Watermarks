from openai import OpenAI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
from nltk.corpus import wordnet

class AdaptiveAttack(BaseModel):
    watermark: int
    paraphrase: str

    def to_dict(self):
        return self.model_dump()

class Evaluation:
    def __init__(self, config: dict) -> None:
        self.model = config['model']
        self.api_key = config['api_key']
        self.client = OpenAI(api_key=config['api_key'])

    def paraphrase(self, text: str) -> str:

        PROMPT = f'''
        You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. Do not leave out any important details when rewriting in your own voice. This is the text: {text}
        '''

        message = [
            {"role": "user", "content": PROMPT}
        ]

        response = self.client.responses.create(
                model=self.model,
                input=message,
            )
        return response.output_text
    
    def adaptive_attack(self, text: str) -> str:
        PROMPT = f'''
        The following text may be watermarked using a certain method. Please first identify if it is watermarked, then, if it is, try to paraphrase the text to remove the watermark. If it is not watermarked, also paraphrase the text. Ensure that the final output contains the same information as the original text and has roughly the same length. Do not leave out any important details when rewriting in your own voice. Output the result in the JSON format with nothing else: {{ "watermark": <0 or 1>, "paraphrase": <the paraphrased text> }}.
        
        This is the text: {text}
        '''

        message = [
            {"role": "user", "content": PROMPT}
        ]

        response = self.client.responses.parse(
                model=self.model,
                input=message,
                text_format=AdaptiveAttack,
            )
        return response.output_parsed
    
    def get_synonym(self, word):
        synsets = wordnet.synsets(word)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return random.choice(list(synonyms)) if synonyms else word

    def random_word_replacement(self, text, p=0.3, seed=42):
        ''' Randomly replace words in the text with their synonyms. 
            Only replaces Nouns, verbs, adjectives, and adverbs.
        '''


        words = word_tokenize(text)

        random.seed(seed)  # Set a seed for reproducibility
        num_to_replace = int(len(words) * p)
        indices = random.sample(range(len(words)), num_to_replace)
        
        new_words = words.copy()
        replaced_count = 0

        for idx in indices:
            if words[idx].isalpha(): # Check if the word is alphabetic
                synonym = self.get_synonym(words[idx])
                if synonym != words[idx]:
                    new_words[idx] = synonym
                    replaced_count += 1

        new_text = ' '.join(new_words)
        return new_text

    def random_word_deletion(self, text: str, deletion_ratio: float = 0.3, seed: int = 42) -> str:
        # Tokenize into words/punctuation
        tokens = word_tokenize(text)
        
        # Determine how many tokens to delete
        num_to_delete = int(len(tokens) * deletion_ratio)
        
        # Optionally seed randomness
        if seed is not None:
            random.seed(seed)
        
        # Randomly pick indices to delete
        delete_indices = set(random.sample(range(len(tokens)), num_to_delete))
        
        # Build new token list, skipping deleted indices
        new_tokens = [tok for i, tok in enumerate(tokens) if i not in delete_indices]
        
        # Detokenize back into a string
        detok = TreebankWordDetokenizer().detokenize(new_tokens)
        return detok