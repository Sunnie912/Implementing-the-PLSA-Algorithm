import numpy as np
import math
import re


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix
       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        with open(self.documents_path, "r") as path:
            lines = path.readlines()
            for line in lines:
                document = []
                document = re.split("\t|\n| ", line)
                self.documents.append(document)
        self.number_of_documents = len(self.documents)

#        path.close()
        #print(self.number_of_documents)

#        print(self.documents_path)
#        with open(self.documents_path, 'r') as file:
#            for line in file.readlines():
#                doc = list()
#                doc.extend(line.split())
#                self.documents.append(doc)
#                self.number_of_documents += 1
#
#        # print(self.documents)
#        print(len(self.documents))
#        print(self.number_of_documents)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]
        Update self.vocabulary_size
        """
        # #############################
        # your code here
        for document in self.documents:
            for word in document:
                if word not in self.vocabulary and word != "":
                    self.vocabulary.append(word)
        self.vocabulary_size = len(self.vocabulary)
        print("size1: ", self.vocabulary_size)

        res = set()
        for doc in self.documents:
            res.update(doc)
        self.vocabulary = res
        self.vocabulary_size = len(res)
        print("size2: ", self.vocabulary_size)
        self.vocabulary_dist = {k: i for i, k in enumerate(self.vocabulary)}

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document,
        and each column represents a vocabulary term.
        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
#        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], dtype = np.int64)
#        for index_doc, document in enumerate(self.documents):
#            term_count = np.zeros([self.vocabulary_size])
#            for word in document:
#                if word in self.vocabulary:
#                    index_term = self.vocabulary.index(word)
#                    term_count[index_term] +=1
#            self.term_doc_matrix[index_doc] = term_count
##         print(self.term_doc_matrix)

        self.term_doc_matrix = np.zeros(shape=(self.number_of_documents, self.vocabulary_size))

        for i, doc in enumerate(self.documents):
            for term in doc:
                self.term_doc_matrix[i][self.vocabulary_dist[term]] += 1
        # print(self.term_doc_matrix)


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob
        Don't forget to normalize!
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        self.document_topic_prob = np.random.rand(self.number_of_documents,number_of_topics)
        # normalize
        self.document_topic_prob = normalize(self.document_topic_prob)
        
        self.topic_word_prob = np.random.rand(number_of_topics, self.vocabulary_size)
        # normalize
        self.topic_word_prob = normalize(self.topic_word_prob)

        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.
        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self, number_of_topics):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        for index_doc in range(len(self.documents)):
            for index_word in range(len(self.vocabulary)):
                denomitor = 0
                for index_topic in range(number_of_topics):
                    self.topic_prob[index_doc, index_topic, index_word] = self.document_topic_prob[index_doc, index_topic] * self.topic_word_prob[index_topic, index_word]
                    denomitor += self.topic_prob[index_doc, index_topic, index_word]
                for index_topic in range(number_of_topics):
                    self.topic_prob[index_doc, index_topic, index_word] /= denomitor

#        self.topic_word_prob = np.nan_to_num(self.topic_word_prob)
#        for doc in range(self.topic_prob.shape[0]):
#            for voc in range(self.topic_prob.shape[2]):
#                self.topic_prob[doc, :, voc] = self.document_topic_prob[doc, :] * self.topic_word_prob[:, voc]
#                self.topic_prob[doc, :, voc] /= self.topic_prob[doc, :, voc].sum()
#        self.topic_word_prob = np.nan_to_num(self.topic_word_prob)
        
    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        for index_topic in range(number_of_topics):
            for index_word in range(len(self.vocabulary)):
                for index_doc in range(len(self.documents)):
                    count = self.term_doc_matrix[index_doc, index_word]
                    self.topic_word_prob[index_topic, index_word] += count * self.topic_prob[index_doc, index_topic, index_word]
            self.topic_word_prob[index_topic, :] /= self.topic_word_prob[index_topic, :].sum()
            
#        for topic in range(self.topic_prob.shape[1]):
#            for voc in range(self.topic_prob.shape[2]):
#                self.topic_word_prob[topic, voc] = self.term_doc_matrix[:, voc].dot(self.topic_prob[:, topic, voc])
#            self.topic_word_prob[topic, :] /= self.topic_word_prob[topic, :].sum()
#        self.topic_word_prob = np.nan_to_num(self.topic_word_prob)

        # update P(z | d)  Pi

        # ############################
        # your code here
        for index_doc in range(len(self.documents)):
            for index_topic in range(number_of_topics):
                for index_word in range(len(self.vocabulary)):
                    count = self.term_doc_matrix[index_doc, index_word]
                    self.document_topic_prob[index_doc, index_topic] += count * self.topic_prob[index_doc, index_topic, index_word]
            self.document_topic_prob[index_doc, :] /= self.document_topic_prob[index_doc, :].sum()

#        for doc in range(self.topic_prob.shape[0]):
#            for topic in range(self.topic_prob.shape[1]):
#                self.document_topic_prob[doc, topic] = self.term_doc_matrix[doc, :].dot(self.topic_prob[doc, topic, :])
#            self.document_topic_prob[doc, :] /= self.document_topic_prob[doc, :].sum()
#        self.document_topic_prob = np.nan_to_num(self.document_topic_prob)

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods
        """
        # ############################
        # your code here
        likelihood = 0.0
        for index_doc in range(len(self.documents)):
            sum = 0
            for index_word in range(len(self.vocabulary)):
                sum1 = 0;
                for index_topic in range(number_of_topics):
                    sum1 += self.document_topic_prob[index_doc,index_topic]*self.topic_word_prob[index_topic,index_word]
                sum1 = np.log(sum1)
            sum += self.term_doc_matrix[index_doc, index_word] * sum1
        likelihood += sum
         
        
#         self.likelihoods.append(np.sum(np.log(self.document_topic_prob @ self.topic_word_prob) * self.term_doc_matrix))
#        likelihood = np.sum(self.term_doc_matrix * np.log(np.matmul(self.document_topic_prob, self.topic_word_prob)))
        self.likelihoods.append(likelihood)
        
        return self.likelihoods[-1]

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0
        last_topic_prob = self.topic_prob.copy()
        
        for iteration in range(max_iter):
            
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            self.expectation_step(number_of_topics)
            diff = abs(self.topic_prob - last_topic_prob)
            L1 = diff.sum()
            print ("L1: ", L1)
            print (last_topic_prob)
            # assert L1 > 0
            last_topic_prob = self.topic_prob.copy()

            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            tmp_likelihood = self.calculate_likelihood(number_of_topics)
            if iteration > 100 and abs(current_likelihood - tmp_likelihood) < epsilon/10:
                print('Stopping', tmp_likelihood)
                return tmp_likelihood
            current_likelihood = tmp_likelihood
            print(max(self.likelihoods))
            
#            self.maximization_step(number_of_topics)
#            self.calculate_likelihood(number_of_topics)
#
#            gap = np.abs(self.calculate_likelihood(number_of_topics) - current_likelihood)
#
#            if gap < epsilon:
#                break;
#            else:
#                current_likelihood = self.calculate_likelihood(number_of_topics)
#
#        return self.topic_word_prob, self.document_topic_prob
            
            
            

def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
