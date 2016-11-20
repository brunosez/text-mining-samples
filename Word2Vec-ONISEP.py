#
# -*- coding: latin-1 -*-
#
from nltk.tokenize import RegexpTokenizer
import numpy as np
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

from gensim.models import Word2Vec

import gensim
from glob import glob
import sys
import io
import argparse

import Stemmer

from sklearn.neighbors import NearestNeighbors



# nettoyage des stop words
# et stemming
#
def token_and_stem(i):

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in fr_stop]
#    print("stopped tokens = ",stopped_tokens)
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stemWord(i) for i in stopped_tokens]
    # stem(i) with PorterStemmer
    #print("stemmed tokens = ", stemmed_tokens)
    # add tokens to list
    return stemmed_tokens

# pour une liste de mots donne la moyenne des word2vec de chaque mots
#
def makeFeatureVec(words, model, num_features, vocab):
    words = [w for w in words if w in vocab]
    featureVec = np.mean([model[w] for w in words],axis=0)
    return featureVec



#
# ESSAI Word2Vec Pour l'ONISEP
#

parser = argparse.ArgumentParser(description='Implémentation Word2Vec: python Word2Vec-ONISEP.py "rep/*.txt", où rep est un répertoire contenant tous les documents à analyser  --nb_features ')
parser.add_argument('input_dir', help = 'répertoire en entrée')
parser.add_argument('--nb_features',type=int, help = 'Nbre de features (dimension) de l espace Word2Vec')


args = parser.parse_args()


input_dir = args.input_dir

#input_dir = sys.argv[1]

if args.nb_features:
  num_features = args.nb_features
else:
  num_features = 300


tokenizer = RegexpTokenizer(r'\w+')

# create French stop words list
fr_stop = get_stop_words('fr')

# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()

print Stemmer.algorithms()
p_stemmer =  Stemmer.Stemmer('french')
    
# create sample documents
# compile sample documents into a list
doc_set = []

# doc pour tester les topics d'un doc
doc_w2v = []
doc_titre_w2v = []

for i, glob_file in enumerate( glob(input_dir) ):
      print "parsing:", glob_file
      # sort glob_file by first column, ignoring the first line
      lines = io.open(glob_file,encoding='utf8').read()
      doc_set.append(lines)

      # Pour test un doc sur 10/3 va être analysé vis à vis des topics calculés
      #
      if i % 3 == 0 :
        doc_w2v.append(lines)
        doc_titre_w2v.append(glob_file)
      # close stream ?



# list for tokenized documents in loop
texts = []

# loop through document list




for i in doc_set:
    
    texts.append(token_and_stem(i))


# Set values for various parameters
# for Word2Vec model
#num_features = 300    # Word vector dimensionality

min_word_count = 4    # Minimum word count
num_workers = 6       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
# Sentences = list of words
# So we concatenate all the texts
sentences = texts
#print("Debug texts =", sentences)
#model = Word2Vec(sentences, workers=num_workers, \
#            size=num_features, min_count = min_word_count, \
#            window = context, sample = downsampling, seed=1)

# workers need cython installed
model = Word2Vec(sentences, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_4minwords_6context"
ddir = "./models"
model.save(ddir+model_name)

#model.doesnt_match("man woman child kitchen".split())
#model.doesnt_match("france england germany berlin".split())
#model.doesnt_match("paris berlin london austria".split())
try:
  print ("test absent = ",model.most_similar("absent"))
except:
  print ("not found in vocab")
#model.most_similar("queen")
#model.most_similar("awful")


# pour un doc donne on donne ses "topics"
# get a word2vec means for a doc

vocab  = set(model.index2word)
print(" vocab =", vocab)

# END Word2Vec model generation
#
######################################################################################

 ## ldamodel.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
##Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
  #print (" test doc 2 word2vec", featureVec)

#######################################################################################

##
# Need to read the Thesaurus as vector
# and run KNN between doc and thesaurus cloud.

lines = io.open('Thesaurus.skos.csv',encoding='utf8').readlines()

not_in_vocab = 0
num_line = 0

## Liste des mots contenus dans le modele Word2Vec
#
vocab_set = set([])
for l in lines:
  #print("in Thesaurus line = ", l)
  # Parsing simple on prend le mot en ""
  # "grèce"@fr va donner grèce
  # mais "secteur agriculture"@fr donnera secteur agriculture et ne sera pas trouvé dans le vocabulaire, donc deuxieme split

  num_line += 1
  liste_complete = l.split('"')
  #print("in Thesaurus word = ", liste_complete[1])
  liste_mots = liste_complete[1].split(' ')

  #####[i for i in tokens if not i in fr_stop]

#  liste_mots_filtre = [ mot for mot in liste_mots if not mot in fr_stop]
#
# Compare stop_word list with Nicolas Gaude's one
#
  liste_mots_filtre = [ mot for mot in liste_mots if ((not mot in fr_stop) and (len(mot) > 1))]

  for mot in  liste_mots_filtre:
    try:
     # ret = model.most_similar(mot)
      ret = model.most_similar( p_stemmer.stemWord(mot))
     # print("in Thesaurus word found = ", mot , ret)
      vocab_set.add(p_stemmer.stemWord(mot))
    except:
      not_in_vocab += 1


print ("Thesaurus stats n_words, n_not in vocabulary, taille vocabulaire trouvé", num_line, not_in_vocab, len(vocab_set))

# Parcours du set, pour construire
# liste des mots , matrice des coordonnées dans l'espace du Model W2Vec
i = 0
X = np.zeros((len(vocab_set),num_features),dtype="float32") 

liste_mots = list(vocab_set)

for elem in liste_mots:
  ##print(" elem, i =", elem, i)
 # liste_mots[i] = elem
  X[i,] = model[elem]
  i += 1

X.reshape(-1, 1)

##print("X =", X[:2,])

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
###############################################

# KNN Algo pour chaque doc
#
doc_num = 1

outfile = io.open("parsing-Word2Vec-docs.csv", mode='w', encoding='utf8')

u = unicode("id_doc,doc_titre,proche1,proche2,proche3,proche4,proche5\n", "utf-8")

outfile.write(u)
#
for i in doc_w2v:
  text = token_and_stem(i)
  try:
    featureVec = makeFeatureVec( text, model, num_features, vocab)
    distances, indices = nbrs.kneighbors(featureVec.reshape(1,-1))
  except:
    print("Exception value error KNN")
  
  line ="%d,%s,"%(doc_num,doc_titre_w2v[doc_num -1])
  for j in indices[0]:
    print(" Doc Num , Plus proches = ", doc_num, liste_mots[j])
    line +="%s,"%liste_mots[j]
  line +="\n"
  outfile.write(line)
  doc_num +=1
  print("")
