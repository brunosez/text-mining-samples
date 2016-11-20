#
# -*- coding: latin-1 -*-
#
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from glob import glob
import sys
import io
import argparse

import Stemmer

#stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')



#
# ESSAI LDA Pour l'ONISEP
#
# Argument entrée : juste le repertoire avec les fichiers à parser
#
# en sortie 2 fichiers csv 

# 1°) parsing-LDA-topics.csv)

# Avec les colonnes : id_topic,mot1,p1,mot2,p2,mot3,p3,mot4,p4,mot5,p5

# TBD Ajuster les 5 topics

# 2°) parsing-LDA-docs.csv

# Avec les colonnes en nbre variable id_doc,topics,topics_probas

# Repertoire officiel est out_input, qui parse les fcihiers input du repo git

class Topic:
    def __init__(self, num, string, liste_mots, liste_probas):
        self.num = num
        self.string = string
        self.liste_mots = liste_mots
        self.liste_probas = liste_probas

    def __setitem__(self, key, value):
        self.values[key] = value

liste1 = list({"ecole", "maitre"})

liste2 = list({0.1, 0.9})

myTopic = Topic(1,"1-ecole-maitre", liste1, liste2 )

#print myTopic.num, myTopic.string , myTopic.liste_mots, myTopic.liste_probas

parser = argparse.ArgumentParser(description='Implémentation LDA : python LDA-ONISEP.py "rep/*.txt", où rep est un répertoire contenant tous les documents à analyser ')
parser.add_argument('input_dir', help = 'répertoire en entrée')
parser.add_argument('--nb_topics',type=int, help = 'Nbre de topics pour LDA')
parser.add_argument('--nb_words', type=int, help = 'Nbre de mots pour LDA')


args = parser.parse_args()


input_dir = args.input_dir

if args.nb_words:
  nb_words = args.nb_words
else:
  nb_words = 4

if args.nb_topics:
  nb_topics = args.nb_topics
else:
  nb_topics = 5



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
doc_lda = []
doc_titre_lda = []

for i, glob_file in enumerate( glob(input_dir) ):
      #print "parsing:", glob_file
      # sort glob_file by first column, ignoring the first line
      lines = io.open(glob_file,encoding='utf8').read()
      doc_set.append(lines)

      # Pour test un doc sur 5 va être analysé vis à vis des topics calculés
      #
     # if i % 2 == 0 :
      doc_lda.append(lines)
      doc_titre_lda.append(glob_file)
      # close stream ?

#list_topics = Topic[nb_topics]
list_topics = {}

def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    ## topics = lda.show_topics(topics=-1, topn=n, formatted=False)

    topics = lda.show_topics(num_topics=nb_topics, num_words=nb_words, formatted=False)

    
    for ti, topic in enumerate(topics):
        print (" ti, topic =", ti, topic)
        #print 'my topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic))

        #topic = Topic( ti+1, string, liste_mots, liste_probas)
        #list_topics.add(topic)



# list for tokenized documents in loop
texts = []

# loop through document list

def token_and_stem(i):

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    # filtre sur stop word et mot de 2 caracteres ou plus
    stopped_tokens = [i for i in tokens if ((not i in fr_stop) and (len(i) > 1)) ]
#    print("stopped tokens = ",stopped_tokens)
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stemWord(i) for i in stopped_tokens]
    # stem(i) with PorterStemmer
    #print("stemmed tokens = ", stemmed_tokens)
    # add tokens to list
    return stemmed_tokens


for i in doc_set:
    
    texts.append(token_and_stem(i))

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
# TBD pur num-topics and passes in arguments
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=500)


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=nb_topics, id2word = dictionary, passes=1000)


#print(ldamodel.print_topics(num_topics=10, num_words=6))

print(ldamodel.show_topics(num_topics=nb_topics, num_words=nb_words, formatted=True))

print_topics( ldamodel, dictionary , nb_words)


outfile_topic = io.open("parsing-LDA-topics.csv", mode='w', encoding='utf8')

u = unicode("id_topic,mot1,p1,mot2,p2,mot3,p3,mot4,p4,mot5,p5\n", "utf-8")

outfile_topic.write(u)

i = 0

# Question un-stemming , 
#import pattern.en as en
#base_form = en.lemma('ate') # base_form == "eat"

#No, there isn't. With stemming, you lose information, not only about the word form (as in eat vs. eats or eaten), 
#but also about the word itself (as in tradition vs. traditional). 
#Unless you're going to use a prediction method to try and predict this information on the basis of the context of the word, there's no way to get it back.

lda_dict = ldamodel.show_topics(num_topics=nb_topics,num_words=nb_words, formatted=False)

print("")
liste_mots=range(nb_words)
liste_probas=range(nb_words)

liste_topics = []

for topic in lda_dict:
        i = i + 1
        #line ="%d,"%i
        line =","
        print "Topic #" + str(i) + ":", topic
        j = 0
        topic_string = ""
        
        for p, id in topic[1]:
            print p, id
            j += 1
           # print("debug j =", j)
            liste_mots[ j - 1 ] = p
            liste_probas[ j - 1 ] = id

            if j == len(topic[1]) :
              #line += "%s,%.3f\n"%(p,id)
              line += "%s,%.3f"%(p,id)
              topic_string += p
            else:
              line += "%s,%.3f,"%(p,id)
              topic_string += p+"-"
        print("topic-string, listes = ", topic_string, liste_mots, liste_probas)
        liste_topics.append (Topic(i,topic_string,liste_mots,liste_probas))
        #u =  unicode(line,"utf-8")  
        #topic = Topic(i,)
        line =str(i)+"="+topic_string+line+"\n"
        outfile_topic.write(line)
        print ""
       # outfile_topic.write
# vocab ?
# vocab = vec.get_feature_names()
#
#print_topics(ldamodel, corpus)

# pour un doc donne on donne ses "topics"

outfile = io.open("parsing-LDA-docs.csv", mode='w', encoding='utf8')

u = unicode("id_doc,doc_titre,topics,topics_probas\n", "utf-8")

outfile.write(u)

doc_num = 0

for i in doc_lda:
  doc_num += 1
  text = token_and_stem(i)
  dictionary = corpora.Dictionary([token_and_stem(i)])
  bow = dictionary.doc2bow(text)

  ret =ldamodel.get_document_topics(bow)
 # argument per word topic KO ??
 ## ret =ldamodel.get_document_topics(bow,per_word_topics=True)

 ## ldamodel.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
##Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
  #print (" test lda to doc", ret)
  line ="%d,"%doc_num
  line += doc_titre_lda[doc_num - 1]+","
  topic_num = 0
  for k in ret:

    line +="%d="%int(k[0]+1)
    topic_num += 1
    #print(" topic num, string", k[0], liste_topics[k[0]])
    myTopic = liste_topics[k[0]]
    #print( " topic_string =", myTopic.string)
    line += myTopic.string+","
    if topic_num == len(ret):

      line += "%.3f\n"%(k[1])
    else:
      line += "%.3f,"%(k[1])
#
# Numero de topic recalé de 1 à N au lieu 0 à N-1
#
    print "Topic #%d "%(int(k[0])+1)
    print "Proba %0.3f"%k[1]
    

    

 #   print ("line =", line)
        #u =  unicode(line,"utf-8")  
        #outfile_topic.write(line)
    print ""

  #u = unicode(line , "utf-8")
  outfile. write(line)
  



