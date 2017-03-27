
import string
from spacy.en import English
global parser
parser = English()
from nltk.corpus import stopwords as stopwords


def cleanPassage(rawtext):
    global parser

    #some code from https://nicschrading.com/project/Intro-to-NLP-with-spaCy/

    #if data is bad, return empty
    if type(rawtext) is not str:
        return ''
    
    #split text with punctuation
    bad_chars = "".join(string.punctuation)
    for c in bad_chars: rawtext = rawtext.replace(c, "")
    
    #parse 
    tokens = parser(rawtext)

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    
    return tokens




def getLemmas(tokens):
    # lemmatize
    lemmas = [tok.lemma_.lower().strip() for tok in tokens]
    return lemmas


def makeNodelist(tokens,probs_cutoff_lower,limitPOS=None):
#    BADPOS = ['PUNCT','NUM','X','SPACE']
    if limitPOS:
        GOODPOS = limitPOS
    else:
        GOODPOS = ['NOUN','PROPN','VERB','ADJ','ADV']
    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
    probs_cutoff_upper = -7.6 #by inspection of sample data
    nodes = []
    lemmas = []
    for tok in tokens:
        goodPOS = tok.pos_ in GOODPOS 
        notStopword = tok.orth_ not in stopwords.words('english')
        notSymbol = tok.orth_ not in SYMBOLS
        isMeaningful = tok.prob > probs_cutoff_lower and tok.prob < probs_cutoff_upper
        
        if goodPOS and notStopword and notSymbol and isMeaningful:
            nodes.append(tok.lemma_+' '+tok.pos_)
            lemmas.append(tok.lemma_)
    return lemmas  

def findMeaningfulCutoffProbability(alltokens):
    probs = [tok.prob for tok in alltokens]
    #set probs_cutoff by inspection by looking for the elbow on the plot of sorted log probabilities
#    probs_cutoff = 500
#    probs_cutoff = probs[int(input("By inspection, at which rank is the elbow for the log probability plot? [integer]"))]
    
    #removing the lowest observed probability seems to remove most of the spelling errors
    probs_cutoff_lower = min(probs)
    return probs_cutoff_lower

def david_dict(data):
    cats = [d[0] for d in data if "noise" not in d[0]]
    dout = {c:set([int(d[1]) for d in data if d[0]==c]) for c in cats}
    #dout['noise'] = set([int(d[1]) for d in data if 'noise' in d[0]])
    return dout

