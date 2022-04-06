from sklearn.decomposition import NMF
import numpy as np
import sys
from timeit import default_timer as timer
from numpy import linalg as LA
import scipy.sparse as sps
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy
from numpy.core.umath_tests import inner1d
import copy

REDUCE_PREC=False   # doesn't work as timereg can be huge. dtype of matrices can get upcasted during operations

def compute_vocab_stats(filename):
    cf=dict()
    df=dict()
    fp=open(filename,'r')
    for line in fp:
        text=line.strip().lower()
        tokens=text.split(' ')
        for w in tokens:
            if w not in cf:
                cf[w]=0
            cf[w]+=1
        for w in set(tokens):
            if w not in df:
                df[w]=0
            df[w]+=1    
    fp.close()
    return cf, df

# rm_top may end up removing some informative terms
# combines the vocab from previous & current timestep, then sort by count freq & filters
# Excess terms beyond vocab size are filtered based on count freq
def clean_sliding_corpus(filename, prev_cf, prev_df, min_cf:int=2, min_df:int=2, rm_top:int=0, max_vocab_size:int=80000):
    curr_cf, curr_df=compute_vocab_stats(filename)
    cf=copy.copy(curr_cf)
    for w, freq in curr_cf.items():
        if w in prev_cf:
            cf[w] = freq + prev_cf[w]   # w in both prev and current
    for w, freq in prev_cf.items():
        if w not in curr_cf:  # w in prev only
            cf[w]=freq
            
    df=copy.copy(curr_df)
    for w, freq in curr_df.items():
        if w in prev_df:
            df[w] = freq + prev_df[w]   # w in both prev and current
    for w, freq in prev_df.items():
        if w not in curr_df:  # w in prev only
            df[w]=freq
            
    temp=[]
    for w, freq in cf.items():
        temp.append([w, freq])
    sorted_cf=sorted(temp, key=lambda x:x[1], reverse=True)  # descending
    exclusion=set()
    for i in range(0, rm_top):
        w=sorted_cf[i][0]
        exclusion.add(w)
        
    for w, freq in sorted_cf:
        if freq < min_cf:
            exclusion.add(w)
    for w, freq in df.items():
        if freq < min_df:
            exclusion.add(w)
            
    vocab=set()
    if max_vocab_size==-1:  # no bound
        for w in cf:
            if w not in exclusion:
                vocab.add(w)
    else:
        for w, f in sorted_cf:
            if w not in exclusion:
                vocab.add(w)
            if len(vocab)==max_vocab_size:
                break
    print('vocab', len(vocab))
    
    cleaned_corpus=[]
    doc_cnt=0
    fp=open(filename,'r')
    for line in fp:
        text=line.strip().lower()
        tokens=text.split(' ')
        filtered_tokens=[]
        for w in tokens:
            if w in vocab:
                filtered_tokens.append(w)
        if len(filtered_tokens)>0:
            cleaned_corpus.append(filtered_tokens)
            doc_cnt+=1
    fp.close()
    print('doc_cnt',doc_cnt)
    return cf, df, vocab, cleaned_corpus
            
# words with count freq < min_cf are removed
# words with doc freq < min_df are removed
# rm_top: no. of top freq words to remove
# numbers not removed. some numbers are meaningful            
def clean_corpus(filename, min_cf:int=2, min_df:int=2, rm_top:int=0):
    cf, df=compute_vocab_stats(filename)
    temp=[]
    for w, freq in cf.items():
        temp.append([w, freq])
    sorted_cf=sorted(temp, key=lambda x:x[1], reverse=True)  # descending
    exclusion=set()
    for i in range(0, rm_top):
        w=sorted_cf[i][0]
        exclusion.add(w)
        
    for w, freq in sorted_cf:
        if freq < min_cf:
            exclusion.add(w)
    for w, freq in df.items():
        if freq < min_df:
            exclusion.add(w)
    
    vocab=set()
    for w in cf:
        if w not in exclusion:
            vocab.add(w)
    cleaned_corpus=[]
    doc_cnt=0
    fp=open(filename,'r')
    for line in fp:
        text=line.strip().lower()
        tokens=text.split(' ')
        filtered_tokens=[]
        for w in tokens:
            if w in vocab:
                filtered_tokens.append(w)
        if len(filtered_tokens)>0:
            cleaned_corpus.append(filtered_tokens)
            doc_cnt+=1
    fp.close()
    print('doc_cnt',doc_cnt)
    return cf, df, cleaned_corpus


# equialent to trace(A' * B)
def tr(A, B):
    elementprod=np.multiply(A, B)
    value=np.sum(elementprod)
    return value

def demo():
    A=np.random.rand(10)
    print(tr(A,A))
    fro_norm=LA.norm(A)
    print(np.power(fro_norm,2))

# very expensive. adds a few hundred % to compute time
# timereg*trace(I) omitted as it's a constant
# omitted constants
#   timereg*trace(I)
#   trXX
def compute_loss_slow(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    WH = np.matmul(W, H)
    WMR = np.matmul(W, MR)
    tr1 = - 2*tr(X,WH) + tr(WH,WH)
    tr2 = - 2*tr(X,WMR) + tr(WMR,WMR)
    tr3 = timereg * (tr(M,M) - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4
    return loss

# Huge matrices not computed since we only want trace
# sequence of matrix multiplication is designed s.t. only smaller matrices are in memory
# constant tr(X'X) can be excluded. Include to get exact value with naive compute
def compute_loss(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    if scipy.sparse.issparse(X):
        norm_X=sparse.linalg.norm(X)
    else:
        norm_X=LA.norm(X)
    TrXX = norm_X * norm_X  #tr(X'X)
    constant=2*TrXX

    XtW = (X.transpose()).dot(W)    # V x K
    tr1_1=np.sum(inner1d(XtW, H.T))
    Ht=H.transpose()
    WtW=(W.T).dot(W)    # K x K
    HtWtW=Ht.dot(WtW)   # V x K
    tr1_2=np.sum(inner1d(HtWtW, Ht))    # trace(H'W'WH)
    tr1=-2*tr1_1 + tr1_2

    XtWM=np.matmul(XtW, M)
    tr2_1=np.sum(inner1d(XtWM, R.T))
    RtMtWtW=np.matmul(MR.T, WtW)
    tr2_2=np.sum(inner1d(RtMtWtW, MR.T))    # trace(R'M'W'WMR)
    tr2=-2*tr2_1 + tr2_2

    norm_M = LA.norm(M)
    trMM = norm_M * norm_M
    tr3 = timereg * (trMM - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4 + constant
    return loss
 
# omitted constants: timereg*trace(I)
# included constants: Tr(X'X)
# 35% faster than slow version
# mem intensive cos WH, WMR matrics are not sparse
def compute_loss_naive(X, W, H, M, R, l1reg, timereg):
    MR = np.matmul(M, R)
    WH = np.matmul(W, H)    # huge
    WMR = np.matmul(W, MR)  # huge
    tr1_norm = LA.norm(X-WH)
    tr1 = tr1_norm * tr1_norm
    tr2_norm = LA.norm(X-WMR)
    tr2 = tr2_norm * tr2_norm
   
    '''   
    # for checking: loss values will sync with the slow version / mem-light version ==
    norm_X = LA.norm(X)
    TrXX = norm_X * norm_X
    tr1 -= TrXX
    tr2 -= TrXX
    '''

    norm_M = LA.norm(M)
    trMM = norm_M * norm_M
    tr3 = timereg * (trMM - 2*np.trace(M))
    tr4 = l1reg * (np.sum(H) + np.sum(W) + np.sum(M))
    loss = tr1 + tr2 + tr3 + tr4
    return loss

def nmf_loss_naive(X, W, H, l1reg):
    WH = np.matmul(W, H)
    tr1_norm = LA.norm(X-WH)
    tr1 = tr1_norm * tr1_norm
    loss = tr1 + l1reg *(np.sum(H)+np.sum(W))
    return loss

def nmf_loss(X, W, H, l1reg):
    XtW = (X.transpose()).dot(W)    # V x K
    tr1_1=np.sum(inner1d(XtW, H.T))
    Ht=H.transpose()
    WtW=(W.T).dot(W)    # K x K
    HtWtW=Ht.dot(WtW)   # V x K
    tr1_2=np.sum(inner1d(HtWtW, Ht))    # trace(H'W'WH)
    tr1=-2*tr1_1 + tr1_2
    loss = tr1 + l1reg *(np.sum(H)+np.sum(W))
    '''
    if scipy.sparse.issparse(X):
        norm_X=sparse.linalg.norm(X)
    else:
        norm_X=LA.norm(X)
    TrXX = norm_X * norm_X # tr(X'X)
    loss+=TrXX
    '''
    return loss


def JAL_NMF(X, K, l1reg, maxIter, computeLoss):    
    (N, V)=np.shape(X)
    W=np.random.rand(N,K)
    H=np.random.rand(K,V)
    
    eps=sys.float_info.epsilon
    for iter in range(0, maxIter):
        # W =  W .* ( X*H' ./ max( WHH'+ lambda, eps) )
        if scipy.sparse.issparse(X):
            numer = X.dot(H.transpose())
        else:
            numer = np.matmul(X, H.transpose())
        temp = np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        W = np.multiply(W, np.divide(numer,denom))
        
        # H = H .* (WtX./max(WtW*H+lambda,eps))
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        
        denom = np.matmul(WtW, H) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        H = np.multiply(H, np.divide(WtX,denom))
        
        if computeLoss==True:
            loss=nmf_loss(X, W, H, l1reg)
            print('iter', iter, 'loss', loss)
        
    return [W, H]

# corpus is list of documents
# each document is list of tokens
def form_sparse_X_corpus(corpus):
    vocab=dict()
    rev_vocab=dict()
    N=len(corpus)
    wid=0
    for tokens in corpus:
        for w in tokens:
            if w not in vocab:
                vocab[w]=wid
                rev_vocab[wid]=w
                wid+=1
    rows=[]
    cols=[]
    values=[]
    doc_id=0
    for tokens in corpus:
        tf=dict()
        for w in tokens:
            wid=vocab[w]
            if wid not in tf:
                tf[wid]=0
            tf[wid]+=1
        for wid, freq in tf.items():
            rows.append(doc_id)
            cols.append(wid)
            values.append(freq)
        doc_id+=1
            
    X=csr_matrix((values, (rows, cols)), shape=(N, len(vocab)))
    return [X, vocab, rev_vocab]


# with predefined vocab
def form_sparse_X_corpus_vocab(corpus, vocab):
    N=len(corpus)
    rows=[]
    cols=[]
    values=[]
    doc_id=0
    documents=[]
    for tokens in corpus:
        tf=dict()
        doc=[]
        for w in tokens:
            if w in vocab:
                wid=vocab[w]
                if wid not in tf:
                    tf[wid]=0
                tf[wid]+=1
                doc.append(wid)
        if len(doc)>0:
            documents.append(doc)
        for wid, freq in tf.items():
            rows.append(doc_id)
            cols.append(wid)
            values.append(freq)
        doc_id+=1
    X=csr_matrix((values, (rows, cols)), shape=(N, len(vocab)))
    return X, documents    

def JPP(corpus, vocab_remap, rev_vocab_remap, R, timereg:float=0.0, l1reg:float=1.0, maxIter:int=200, computeLoss:bool=False, seed:int=0):
    X, documents=form_sparse_X_corpus_vocab(corpus, vocab_remap)
    
    # random init param    
    np.random.seed(seed)
    (N, V)=np.shape(X)
    K=R.shape[0]
    W=np.random.rand(N, K) # doc df over topics
    H=np.random.rand(K, V)  # topic df over words
    M=np.random.rand(K, K)  # topic-topic mapping

    scaled_I = timereg * np.identity(K)    # constants
    eps=sys.float_info.epsilon
    loss_delta=0.01
    loss=float('inf')
    start = timer()
    for iter in range(0, maxIter):
        J=np.matmul(M, R)

        # From Matlab: W = W .* ( X(H'+J') ./ max( W( JJ'+HH'+lambda ), eps) )
        # implemented here: W = W .* ( X(H'+J') ./ max( W(JJ'+HH')+lambda, eps) )
        H_J = np.add(H.transpose(), J.transpose())
        if scipy.sparse.issparse(X):
            numer = X.dot(H_J)
        else:
            numer = np.matmul(X, H_J)
        temp = np.matmul(J, J.transpose()) + np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        W = np.multiply(W, np.divide(numer,denom))

        # M = M .* ( (WtX*R' + alpha*I) ./ max( WtW*M*R*R' + alpha*M+lambda,eps) ); 
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        WtXRt = np.matmul(WtX,R.transpose())
        numer = np.add( WtXRt, scaled_I )
        #np.fill_diagonal(WtXRt, WtXRt.diagonal() + timereg)    # matrix is small -> immaterial
        #numer=WtXRt
        WtWM = np.matmul(WtW,M)
        RRt = np.matmul(R, R.transpose())
        WtWMRRt = np.matmul(WtWM, RRt)      #   WtW*M*R*R'

        denom = np.add(WtWMRRt, timereg*M) + l1reg    # + alpha*M + lambda
        denom = np.maximum(denom, eps)    # ensure non-negativity
        M = np.multiply(M, np.divide(numer,denom))

        # H = H .* (WtX./max(WtW*H+lambda,eps));
        denom = np.matmul(WtW, H) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        H = np.multiply(H, np.divide(WtX,denom))
        if computeLoss==True:
            prev_loss=loss
            loss=compute_loss_naive(X, W, H, M, R, l1reg, timereg)    # expensive to eval
            print(loss)
            if prev_loss - loss < loss_delta:   # reduction in loss < threshold
                print("break at iter", iter)
                break
        
    endtime=timer()
    print('JPP elapsed', endtime-start)
    
    topn=30
    topics=[]
    probs=[]
    for k in range(0, K):
        sorted_wid=H[k,:].argsort()[::-1]
        topic=[]
        prob=[]
        for ii in range(0, topn):
            word=rev_vocab_remap[sorted_wid[ii]]
            word_p=H[k, sorted_wid[ii]]
            topic.append(word)
            prob.append(word_p)
        topics.append(topic)
        probs.append(prob)
            
    return [topics,W,H,M, documents]



# handles both dense, sparse X matrix
# objective function in compute_loss_slow()
def JPP_old(X, R, K, timereg, l1reg, loss_delta, maxIter):
    # random init param    
    (N, V)=np.shape(X)
    W=np.random.rand(N, K) # doc df over topics
    H=np.random.rand(K, V)  # topic df over words
    M=np.random.rand(K, K)  # topic-topic mapping

    scaled_I = timereg * np.identity(K)    # constants
    eps=sys.float_info.epsilon
    loss_delta=0.01
    loss=float('inf')
    start = timer()
    for iter in range(0, maxIter):
        J=np.matmul(M, R)

        # From Matlab: W = W .* ( X(H'+J') ./ max( W( JJ'+HH'+lambda ), eps) )
        # implemented here: W = W .* ( X(H'+J') ./ max( W(JJ'+HH')+lambda, eps) )
        H_J = np.add(H.transpose(), J.transpose())
        if scipy.sparse.issparse(X):
            numer = X.dot(H_J)
        else:
            numer = np.matmul(X, H_J)
        temp = np.matmul(J, J.transpose()) + np.matmul(H, H.transpose()) 
        denom = np.matmul(W, temp) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        W = np.multiply(W, np.divide(numer,denom))

        # M = M .* ( (WtX*R' + alpha*I) ./ max( WtW*M*R*R' + alpha*M+lambda,eps) ); 
        WtW = np.matmul(W.transpose(), W)
        if scipy.sparse.issparse(X):
            XtW = (X.transpose()).dot(W)
            WtX = XtW.transpose()
        else:
            WtX = np.matmul(W.transpose(), X)
        WtXRt = np.matmul(WtX,R.transpose())
        numer = np.add( WtXRt, scaled_I )
        #np.fill_diagonal(WtXRt, WtXRt.diagonal() + timereg)    # matrix is small -> immaterial
        #numer=WtXRt
        WtWM = np.matmul(WtW,M)
        RRt = np.matmul(R, R.transpose())
        WtWMRRt = np.matmul(WtWM, RRt)      #   WtW*M*R*R'

        denom = np.add(WtWMRRt, timereg*M) + l1reg    # + alpha*M + lambda
        denom = np.maximum(denom, eps)    # ensure non-negativity
        M = np.multiply(M, np.divide(numer,denom))

        # H = H .* (WtX./max(WtW*H+lambda,eps));
        denom = np.matmul(WtW, H) + l1reg
        denom = np.maximum(denom, eps)    # ensure non-negativity
        H = np.multiply(H, np.divide(WtX,denom))
        '''              
        prev_loss=loss
        loss=compute_loss_naive(X, W, H, M, R, l1reg, timereg)    # expensive to eval
        print(loss)
        if prev_loss - loss < loss_delta:   # reduction in loss < threshold
            print("break at iter", iter)
            break
        '''
        print(iter)                     
    endtime=timer()
    print('JPP elapsed', endtime-start)
    return [H,W,M]


# akin to merge in clean cases
# optional: list > 2 parents
def report_merge_like(M, K, Xvalue:float=0.7):
    child_scores=np.sum(M, axis=1)
    for k in range(0, K):
        sorted_id=M[k,:].argsort()[::-1]    # for each topic(t+1), sort topics(t)
        s1=M[k, sorted_id[0]]
        s2=M[k, sorted_id[1]]
        top_score=s1+s2
        top_coverage=top_score/child_scores[k]
        if (s1/child_scores[k] < Xvalue) and (top_coverage >= Xvalue):  # 1st condition s.t. not under one-one evolving
            print(sorted_id[0:2], '->', k, top_coverage)
            
# akin to split in clean cases
# optional: list > 2 children
def report_split_like(M, K, Xvalue:float=0.7):
    parent_scores=np.sum(M, axis=0)
    for k in range(0, K):
        sorted_id=M[:,k].argsort()[::-1]    # for each topic(t), sort topics(t+1)
        s1=M[sorted_id[0], k]
        s2=M[sorted_id[1], k]
        top_score=s1+s2
        top_coverage=top_score/parent_scores[k]
        if (s1/parent_scores[k] < Xvalue) and (top_coverage >= Xvalue):  # 1st condition s.t. not under one-one evolving
            print(k, '->', sorted_id[0:2], top_coverage)

# Merging nodes: A,B -> C
#   - tgt: Most of C's weight from A, B where most means > %x
# Splitting nodes: A -> C, D
#   - src: most of A's weight -> C, D
# one-one evolving: A->C
#   - src: most of A's weight -> C
#   - tgt: most of C's weight from A
def evolution_analysis(M, R, H, rev_vocab2):
    np.set_printoptions(precision=3, suppress=True)
    (K, K)=M.shape
    
    topn=30
    prev_topics=[]
    for k in range(0, K):
        sorted_wid=R[k,:].argsort()[::-1]
        topic=[]
        for ii in range(0, topn):
            word=rev_vocab2[sorted_wid[ii]]
            topic.append(word)
        topic.append(topic)
    
    threshold=0.4
    tiny=0.001
    print("======emerging=======")
    emerging=set()
    for k in range(0,K):
        all_low=True
        for j in range(0, K):
            if M[k, j]>=threshold:
                all_low=False
                break
        if all_low==True:
            print('new',k)
            emerging.add(k)
            
    print("======dying=======")        
    dead=set()
    for j in range(0, K):   # column
        all_low=True
        for k in range(0,K):    # row
            if M[k,j]>=threshold:
                all_low=False
                break
        if all_low==True:
            print('dead',j)
            dead.add(j)
            
    print("======one to one=======")       
    # must also exclude emerging, dying topics
    parent_scores=np.sum(M, axis=0)
    child_scores=np.sum(M, axis=1)
    child_parent=dict()
    Xvalue=0.7
    for k in range(0, K):
        if k in emerging:
            continue
        for j in range(0, K):
            if j in dead:
                continue
            if M[k, j] >= Xvalue * child_scores[k]:   # from tgt/child's POV, there is a main src/parent_scores
                child_parent[k]=j
                
    # check from parent pov whether child is dominant                
    for child, parent in child_parent.items():
        if M[child, parent] >= Xvalue * parent_scores[parent]:
            print(parent, '->', child, M[child, parent], M[child, parent]/parent_scores[parent], M[child, parent]/child_scores[child])
            
    print("======merge-like======")
    report_merge_like(M, K)
    print("======split-like======")
    report_split_like(M, K)


def update_R(R, old_vocab, new_vocabSet):
    (K, old_V)=R.shape
    new_R=np.zeros((K, len(new_vocabSet)))
    revmap=dict()
    vocab_map=dict()
    wid=0
    for w, old_wid in old_vocab.items():
        if w in new_vocabSet:
            new_R[:,wid]=np.copy(R[:, old_wid])
            revmap[wid]=w
            vocab_map[w]=wid
            wid+=1
    for w in sorted(list(new_vocabSet)):    # for deterministic order
        if w not in vocab_map:
            revmap[wid]=w
            vocab_map[w]=wid
            wid+=1
    return new_R, vocab_map, revmap
        
def NMF_model(corpus, K:int=10, l1reg:float=1.0, seed:int=0, maxIter:int=200, computeLoss:bool=False, topn:int=30):
    [X, vocab, rev_vocab]=form_sparse_X_corpus(corpus)
    np.random.seed(seed)
    [W, H]=JAL_NMF(X, K, l1reg, maxIter, computeLoss)
    
    topics=[]
    for k in range(0, K):
        sorted_wid=H[k, :].argsort()[::-1]
        topic=[]
        for ii in range(0, topn):
            word=rev_vocab[sorted_wid[ii]]
            topic.append(word)
        topics.append(topic)
    return [topics, W, H, vocab, rev_vocab]
        
def jpp_expt():
    seedvalue=1
    num_topics=30
    cf, df, cleaned_corpus=clean_corpus('sample1.txt')
    
    fp=open('sample1_cleaned.txt','w')  # for inspection
    for filtered_tokens in cleaned_corpus:
        fp.write(' '.join(filtered_tokens)+'\n')
    fp.close()
    
    # First time step is just normal NMF / GNMF
    [topics, document_topic, topic_word, vocab, rev_vocab]=NMF_model(cleaned_corpus, K=num_topics, l1reg=0.10, maxIter=200, computeLoss=False, seed=seedvalue)

    for k in range(0,num_topics):
        print(k,':', topics[k])

    documents=[]
    for tokens in cleaned_corpus:
        doc=[]
        for word in tokens:
            if word in vocab:
                doc.append(vocab[word])
        if len(doc)>0:
            documents.append(doc)
            
    timereg=0.0
    l1reg=0.05
    maxIter=200
    computeLoss=False
    timestep=1
    
    prev_cf=cf.copy()
    prev_df=df.copy()
    snapshots=['sample2.txt','sample3.txt']
    cleaned_filelist=['sample2_cleaned.txt','sample3_cleaned.txt']
    for tt in range(0, 2):
        corpus=snapshots[tt]
        cleaned_file=cleaned_filelist[tt]
        cf, df, new_vocabSet, cleaned_corpus = clean_sliding_corpus(corpus, prev_cf, prev_df)
        
        fp=open(cleaned_file, 'w')      # for inspection
        for filtered_tokens in cleaned_corpus:
            fp.write(' '.join(filtered_tokens)+'\n')
        fp.close()
        
        print('old vocab size', len(vocab))
        print('new vocab size', len(new_vocabSet))

        R, vocab_remap, rev_vocab_remap=update_R(topic_word.copy(), vocab, new_vocabSet)
        
        [topics, W, H, M, documents]=JPP(cleaned_corpus, vocab_remap, rev_vocab_remap, R, timereg, l1reg, maxIter, computeLoss, seed=seedvalue)
        
        for k in range(0,num_topics):
            print(k,':', topics[k])

        evolution_analysis(M, R, H, rev_vocab_remap)
        fname='M'+str(timestep)+'.npy'  # for topic map viz
        with open(fname, 'wb') as fp:
            np.save(fp, M)
        vocab=copy.copy(vocab_remap)    
        rev_vocab=copy.copy(rev_vocab_remap)
        topic_word=H.copy()
        prev_cf=cf.copy()
        prev_df=df.copy()
        timestep+=1
        

jpp_expt()
