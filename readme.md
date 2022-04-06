Dynamic Topic Models with Joint-Past-Present Decomposition
======================================

Usage of API is demonstrated in `jpp_demo.py` which conducts dynamic topic modelling on data split into 3 time intervals (each spanning at least a year). The data is assumed to have been cleaned
of stopwords, urls and punctuations. It is also possible to work with shorter time intervals, e.g. weeks or months. The duration matters less than the data quantity per interval. Importantly, there
should be at least a few thousand documents per interval for topic modelling to give sensible results.

The function `JPP` implements the dynamic topic modelling approach in [1], termed as the Joint-Past-Present Decomposition algorithm. Given at least 2 consecutive snapshots of data corresponding to
time interval t-1 and t, the topics at t are inferred based jointly on the data at t and the topics previously inferred in t-1. The latter can be inferred using NMF or by previous application of
JPP. Dynamic topic modelling can be used for retrospective analysis: partition historical data into time intervals and study how topics evolve across intervals. It can also be used to compute
the topics for a new incoming batch of data conditional on the topics that had been inferred on an older batch. 

Input parameters are:

* **corpus**: The list of documents to be used for topic modelling. Each document is a list of tokens. It is assumed that pre-processing and word filtering has already been done (see next section)
* **vocab_remap**: dict mapping words to numeric id
* **rev_vocab_remap**: dict mapping numeric id to words
* **R**: topic-word matrix from time interval t-1. Each row is a topic with scores for each word. Based on the number of matrix rows, the number of topics is automatically set.
* **timereg**: Temporal regularization. High values forces each topic at t to be mapped to each topic at t-1. (default=0)
* **l1reg**: The L1 regularization parameter (default=1.0)
* **maxIter**: The number of iterations to perform multiplicative updates (default=200)
* **computeLoss**: Set to True to compute and print the loss per iteration.  Useful for checking algorithm convergence. However this incurs computation cost and topic inference takes slightly longer.
* **seed**: The seed which affects the random initialization of document-topic and topic-word matrix at time interval t, and the matrix mapping topics from t-1 to t

Returns the following for time interval t:
* **topics**: List of topics where each topic is a list of top 30 scoring words.
* **W**: Document-topic matrix. Each row is a document with scores for each topic.
* **H**: Topic-word matrix. Each row is a topic with scores for each word
* **M**: Topic mapping matrix representing how topics at t-1 (on columns) influence topics at t (on rows). For examle, a high value on (row i, column j) means that topic j at t-1 has a high
influence on topic i at t.
* **documents**: List of documents where each document is a list of numeric word ids.

Preprocessing and Vocab Handling
-------------------------------
As more time intervals are processed, the number of unique words encountered by the model grows. Without vocab pruning, the vocab size grows indefinitely over time, leading to computation and memory
issues. To overcome this issue, we can constrain the vocab size to be at most equal to some user specified threshold. Since the topics at interval t are conditional on the data at t and the topics
at t-1, we construct the vocab based on both time intervals t-1 and t. If the user specified vocab threshold is exceeded, we prune the vocab by sorting and excluding terms with the lowest frequencies.
After pruning, we update the matrix R (topic-word matrix from time interval t-1) such that columns corresponding to pruned words are dropped. The updated R matrix is then used as an input parameter
for dynamic topic modelling. We expect the pruning and updating to have only a small effect on topic inference since low frequency words usually do not play a big role in defining topics.

`clean_sliding_corpus` implements the logic of combining the vocab from time intervals t-1 and t, sorting by count frequency & filtering off low frequency terms if the vocab threshold is exceeded.
Input parameters are:
* **filename**: Text file of the documents in time interval t. Each line corresponds to a document
* **prev_cf**: Dictionary of count frequencies of terms from time interval t-1. Key=term, value=count frequencies
* **prev_df**: Dictionary of document frequencies of terms from time interval t-1. Key=term, value=document frequencies
* **min_cf**: Minimum count frequency for a term to be retained (default=2)
* **min_df**: Minimum document frequency for a term to be retained (default=2)
* **rm_top**: The no. of top words (based on count freq) to be removed (default=0)
* **max_vocab_size**: The maximum allowed vocab size. If exceeded, words with the lowest count frequencies are pruned until the specified vocab size is met. (default=80000)

Returns:
* **cf**: Dictionary of count frequencies of terms from t and t-1 after preprocessing
* **df**: Dictionary of document frequencies of terms from t and t-1 after preprocessing
* **vocab**: Set of words in the preprocessed vocabulary
* **cleaned_corpus**: The list of cleaned documents at t. Each document is a list of tokens. This will be input to JPP() for topic modelling.

Report of Topic Evolutions
--------------------------
The function `JPP` returns a topic mapping matrix which can be visualized for interpreting the topic evolutions. However each topic at t can be influenced by multiple topics from t-1 while each
topic at t-1 can influence multiple topics at t, hence the relationship is many-to-many which makes manual interpretation very tedious. To avoid this, we provide a utility function that analyzes
the mapping matrix and reports certain topic evolution scenarios that are intuitive. 

The function `evolution_analysis` prints out topic ids matching the following scenarios:
* **emerging**: If all the topics at t-1 consistently have little influence on a topic A at t, then A is an emerging topic.
* **dying**: If topic A at t-1 has little influence on  all topics at t, then A is a dying topic.
* **one-to-one**: If topic A at t-1 directs most of its influence (>= X%) towards topic B at t and B receives most of its influence (>=X%) from A, then there is one-to-one topic evolution from 
A to B
* **merge-like**: If topic A at t receives most its influence from 2 topics B and C from t-1, then B and C may have merged to form A. (Typically the merge is not clean as B and C may substantially
influence other topics at t as well)
* **split-like**: If topic A at t-1 direct most of its influence(>=X%) towards 2 topics B and C at t, then A may have split into B and C. (The split may not be clean as B and C may be influenced
by other topics from t-1 as well)

Currently, we set X% at 70%. While additional conditions can be imposed to report only clean merges and splits, we find that such cases are rare and the report may omit too many topic evolutions.

**Example**

The following examle further illustrates why merging/splitting scenarios may not be clean. In the example, topic A at t-1 directs most of its influence towards topics C and D at t. Concurrently,
topic B at t-1 also influences C. Hence it can be argued that A has split into C and D. However A has also merged with B to form C. Such concurrent evolutions can be confusing to interpret in terms
of clean splits / merges.
	
	(A, t-1) --> (C, t), (D, t)
	(B, t-1) --> (C, t)
	
References
------------
[1] Carmen Vaca, Amin Mantrach, Alejandro Jaimes and Marco Saerens, A time-based collective factorization for topic discovery and monitoring in news, WWW 2014. 	






