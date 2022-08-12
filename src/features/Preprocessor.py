import morfeusz2
import re, string
import pandas as pd
from dataclasses import dataclass
from stop_words import get_stop_words


@dataclass
class Preprocessor:
    """ 
    Class responsible for preprocessing text.
    """
    language: str = 'pl'
    stopwords_file_path:str  = 'polish_stopwords.txt'
    
    

    # need improvement to be more flexible (e.g. preprocess more than one language)
    def __post_init__(self):
        if self.language == 'pl':
            # load Morfeusz
            self.morfeusz = morfeusz2.Morfeusz()
            
            # load stopwords from nltk
            self.stopwords = set(get_stop_words('polish'))
            
            # load stopwords from txt and union with stopwords
            self.stopwords |= set(open(self.stopwords_file_path, 'r', encoding='utf-8').read().splitlines())
            


            

    # need to be implemented better (e.g. load from path)
    def fit(self, dataset: pd.DataFrame):
        """
        Loads dataset from file.
        Args:
            dataset: dataset to load
        """
        self.dataset = dataset
    
    # TODO: add more preprocessing methods
    # preprocessing methods
    def lowercase(self, cols:list):
        """
        Lowercases all words in dataset by given columns.
        Args:
            cols: list of columns to lowercase
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].str.lower()
        
        
    def remove_punctuation(self, cols:list):
        """
        Removes punctuation from dataset by given columns.
        Args:
            cols: list of columns to remove punctuation
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].str.translate(str.maketrans('', '', string.punctuation + '„”“”’‘'))
    
    def remove_stopwords(self, cols:list):
        """
        Removes stopwords from dataset by given columns.
        Args:
            cols: list of columns to remove stopwords
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            # self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if word not in self.stopwords]))
            # self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() 
            #                                                             if word not isinstance(word,None) 
            #                                                             and word not in self.stopwords]))

            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split()
                                                                            if word not in self.stopwords]))

            
    
    def lemmatize(self, cols:list):
        """
        Lemmatizes dataset by given columns.
        Args:
            cols: list of columns to lemmatize
        """

        def pl_text_lemmatizer(text: str) -> str:
            """
            Lemmatizes polish text.
            Args:
                text: text to lemmatize
            Returns:
                lemmatized text
            """
            list_of_words = text.split()
            lemmatized_list_of_words = []

            for word in list_of_words:    
                _, _, interpretation = self.morfeusz.analyse(word)[0]
                lemWord = interpretation[1]
                lemWordStripped = lemWord.split(':', 1)[0].lower()
                lemmatized_list_of_words.append(lemWordStripped)

            lemmatizedText = ' '.join(lemmatized_list_of_words)

            return lemmatizedText
            

        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].apply(lambda x: pl_text_lemmatizer(x))
            
    
    def remove_hashtags(self, cols:list):
        """
        Removes hashtags from dataset by given columns.
        Args:
            cols: list of columns to remove hashtags
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            # both implementations seem to work, haven't checked which one is faster
            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('#')]))
            # self.dataset[col] = self.dataset[col].apply(lambda x: re.sub(r'#[a-zA-Z0-9_]+', '', x))
    
    def remove_mentions(self, cols:list):
        """
        Removes mentions from dataset by given columns.
        Args:
            cols: list of columns to remove mentions
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
        
    def remove_urls(self, cols:list):
        """
        Removes urls from dataset by given columns.
        Args:
            cols: list of columns to remove urls
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('http') and not word.startswith('www.')]))


    def remove_numbers(self, cols:list):
        """
        Removes numbers from dataset by given columns.
        Args:
            cols: list of columns to remove numbers
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if not word.isdigit()]))
    

    # TODO: add more preprocessing methods
    #       check remove_emojis
    def remove_emojis(self, cols:list):
        """
        Removes emojis from dataset by given columns.
        Args:
            cols: list of columns to remove emojis
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')

        for col in cols:
            self.dataset[col] = self.dataset[col].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith(':')]))
    

    def transform(self, cols:list,lowercase=True,remove_punctuation=True,
                                    remove_stopwords=True,lemmatize=True,
                                    remove_hashtags=True,remove_urls=True,
                                    remove_numbers=True,remove_emojis=True,
                                    remove_mentions=True):
        """
        Preprocess dataset by given columns.
        Args:
            cols: list of columns to transform
        
        Returns:
            transformed dataset
        """
        if self.dataset is None:
            raise Exception('Dataset is not loaded.')


        self.lowercase(cols) if lowercase else None
        self.remove_mentions(cols) if remove_mentions else None
        self.remove_hashtags(cols) if remove_hashtags else None
        self.remove_urls(cols) if remove_urls else None
        self.remove_numbers(cols) if remove_numbers else None
        self.remove_emojis(cols) if remove_emojis else None
        self.remove_punctuation(cols) if remove_punctuation else None
        self.lemmatize(cols) if lemmatize else None
        self.remove_stopwords(cols) if remove_stopwords else None

        return self.dataset
    
    def fit_transform(self, dataset: pd.DataFrame):
        """
        Loads dataset from file and preprocesses it.
        Args:
            dataset: dataset to preprocess
        """
        self.fit(dataset)
        return self.transform(dataset.columns)