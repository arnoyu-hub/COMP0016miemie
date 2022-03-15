import time, datetime
import json
import pymongo
import numpy as np

from main.NLP.LDA.LDA import Lda
from main.NLP.PREPROCESSING.module_preprocessor import ModuleCataloguePreprocessor
from main.LOADERS.module_loader import ModuleLoader
from main.MONGODB_PUSHERS.mongodb_pusher import MongoDbPusher

class IheModuleLda(Lda):
    """
        Concrete class for mapping UCL modules to UN IHE (United Nations Sustainable Development Goals) using Latent Dirichlet Allocation. 
        The eta priors can be alterned to guide topic convergence given IHE-specific keywords.
    """

    def __init__(self):
        """
            Initialize state of IheModuleLda with module-catalogue preprocessor, module data loader, module data, list of IHE-specific keywords, 
            number of IHE, text vectorizer and model.
        """
        self.preprocessor = ModuleCataloguePreprocessor()
        self.loader = ModuleLoader()
        self.data = None # module-catalogue dataframe with columns {ModuleID, Description}.
        self.keywords = None # list of IHE-specific keywords.
        self.num_topics = 0
        self.vectorizer = self.get_vectorizer(1, 3, 1, 0.03)
        self.model = None

    def create_eta(self, priors: dict, eta_dictionary: dict) -> np.ndarray:
        """
            Sets the eta hyperparameter as a skewed prior distribution over word weights in each topic.
            IHE-specific keywords are given a greater value, aimed at guiding the topic convergence.
        """
        eta = np.full(shape=(self.num_topics, len(eta_dictionary)), fill_value=1) # topic-term matrix filled with the value 1.
        for keyword, topics in priors.items():
            keyindex = [index for index, term in eta_dictionary.items() if term == keyword]
            if len(keyindex) > 0:
                for topic in topics:
                    eta[topic, keyindex[0]] = 1e6 # increases the A-priori belief on topic keyword probability (for GuidedLDA).
                
        return eta

    def write_results(self, corpus, num_top_words: int, results_file: str) -> None:
        """
            Serializes the perplexity, topic-word and document-topic distributions as a JSON file and pushes the data to MongoDB.
        """
        data = {}

        # Save perplexity.
        data['Perplexity'] = self.model.log_perplexity(corpus)

        # Save topic-word distribution.
        data['Topic Words'] = {}
        for n in range(self.num_topics):
            data['Topic Words'][str(n + 1)] = [self.model.id2word[w]for w, p in self.model.get_topic_terms(n, topn=num_top_words)]

        # Save document-topic distribution.
        data['Document Topics'] = {}
        documents = self.data.Module_ID
        for d, c in zip(documents, corpus):
            doc_topics = ['({}, {:.1%})'.format(topic + 1, pr) for topic, pr in self.model.get_document_topics(c)]
            data['Document Topics'][str(d)] = doc_topics
        
        # Push data to MongoDB and serialize as JSON file.
        MongoDbPusher().module_prediction(data)
        with open(results_file, 'w') as outfile:
            json.dump(data, outfile)

    def display_topic_words(self, num_top_words: int) -> None:
        """
            Prints the topic-word distribution with num_top_words words for each IHE.
        """
        for n in range(self.num_topics):
            print('IHE {}: {}'.format(n + 1, [self.model.id2word[w] for w, p in self.model.get_topic_terms(n, topn=num_top_words)]))

    def display_document_topics(self, corpus) -> None:
        """
            Prints the document-topic distribution for each module in the corpus.
        """
        documents = self.data.Module_ID
        count = 0
        for d, c in zip(documents, corpus):
            if count % 25 == 0:
                doc_topics = ['({}, {:.1%})'.format(topic + 1, pr) for topic, pr in self.model.get_document_topics(c)]
                print('{} {}'.format(d, doc_topics))
            count += 1
    
    def run(self) -> None:
        """
            Initializes IheModLda parameters, trains the model and saves the results.
        """
        ts = time.time()
        startTime = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        # Training parameters.
        num_modules = "MAX"
        # IHE-specific keywords.
        keywords = "main/IHE_KEYWORDS/lda_speciality_keywords.csv"
        passes = 10
        iterations = 400
        chunksize = 6000
        num_top_words = 20

        # IHE results files.
        pyldavis_html = "main/NLP/LDA/IHE_MODULE_RESULTS/pyldavis.html"
        tsne_clusters_html = "main/NLP/LDA/IHE_MODULE_RESULTS/tsne_clusters.html"
        model = "main/NLP/LDA/IHE_MODULE_RESULTS/model.pkl"
        results = "main/NLP/LDA/IHE_MODULE_RESULTS/training_results.json"

        self.load_dataset(num_modules)
        self.load_keywords(keywords)
        self.num_topics = len(self.keywords)
        
        print("Training...")
        corpus = self.train(passes, iterations, chunksize)
        self.display_results(corpus, num_top_words, pyldavis_html, tsne_clusters_html)

        print("Saving results...")
        self.write_results(corpus, num_top_words, results)
        self.push_html_postgre("main/NLP/LDA/IHE_MODULE_RESULTS/pyldavis.html", "main/NLP/LDA/IHE_MODULE_RESULTS/tsne_clusters.html", "ihemod")
        self.serialize(model)
        
        print("Done.")