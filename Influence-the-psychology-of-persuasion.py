import PyPDF2
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import string

import base64

def print_most_common_freq_words(words, n=10):
	fdist = FreqDist(words)
	for word, frequency in fdist.most_common(n):
		print ('{};{}'.format(word, frequency))

def read_book(name):
	file = open(name,'rb')
	book = PyPDF2.PdfFileReader(file)
	return book

def get_text(book, start, end):
	text = ''
	pages = []
	for idx in range(start,end,1):
		page = book.getPage(idx)
		text = page.extractText()
		# text.encode('utf-8')
		pages.append(text)
	return pages

def get_sentences(text):
	return sent_tokenize(text)

def get_words_list(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	# tokens = nltk.word_tokenize(text)
	txt = nltk.Text(tokens)
	words = nltk.tokenize.word_tokenize(text)
	return [w.lower() for w in words]

def remove_stopwords(words):
	stop_words = set(stopwords.words('english'))
	filtered_words = [w for w in words if not w in stop_words]
	filtered_sentence = ' '.join(filtered_words)
	words = nltk.tokenize.word_tokenize(filtered_sentence)
	return words

def remove_punctuations_and_small_words(words, word_len):
	words = [w for w in words if w.isalpha()]
	words = [w for w in words if len(w)>word_len]
	return words
	# tokens = nltk.wordpunct_tokenize(raw)
	#	try:
	#		words = [w.maketrans('','',string.punctuation) for w in words]
	#	except AttributeError,e:
	#		print("Attribute error", e)

def stem_words(words):
	# stemming of words
	porter = PorterStemmer()
	return [porter.stem(w) for w in words]

def lemmatize_words(words):
	wordnet_lemmatizer = WordNetLemmatizer()
	return [wordnet_lemmatizer.lemmatize(w).encode('utf-8') for w in words]

def pos_tagging(text):
	# print nltk.pos_tag(text)
	return nltk.pos_tag(text)

def get_pos_text(word_tag_pairs, pos):
	# print word_tag_pairs
	word_fd = [word for (word, tag) in pos_text if tag in pos]
	#print word_fd
	return set(word_fd)

def get_dict(doc):
	from gensim import corpora
	dictionary = corpora.Dictionary(doc)
	return dictionary

def get_df_matrix(doc_clean, dictionary):
	import gensim
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
	return doc_term_matrix

def fit_lda_model(doc_term_matrix, dictionary, topic_count):
	from gensim.models.ldamodel import LdaModel
	ldamodel = LdaModel(doc_term_matrix, num_topics=topic_count, id2word = dictionary, passes=50)
	return ldamodel

def fit_tfidf_model(corpus):
	from gensim.models import TfidfModel
	model = TfidfModel(corpus)  # fit model
	return model

def fit_hdp_model(doc_term_matrix, dictionary):
	from gensim.models import HdpModel
	hdp = HdpModel(doc_term_matrix, dictionary)
	return hdp

if __name__ == "__main__":
	
	book_name = 'Influence the psychology of persuasion.pdf'
	topics = {
			'chap1':{'start':10,'end':22},\
			'chap2':{'start':23,'end':52},\
			'chap3':{'start':53,'end':96},\
			'chap4':{'start':97,'end':135},\
			'chap5':{'start':136,'end':166},\
			'chap6':{'start':167,'end':187},\
			'chap7':{'start':188,'end':214}\
		}
	'''
	book_name = 'How to Act Like a CEO.pdf'
	topics = { \
		'chap1':{'start':9,'end':48}, \
		'chap2':{'start':49,'end':71}, \
		'chap3':{'start':73,'end':90}, \
		'chap4':{'start':91,'end':112}, \
		'chap5':{'start':113,'end':135}, \
		'chap6':{'start':137,'end':149}, \
		'chap7':{'start':151,'end':170}, \
		'chap8':{'start':171,'end':178}, \
		'chap9':{'start':179,'end':186}, \
		'chap10':{'start':187,'end':201} \
	}
	'''

	chaps = {}

	book = read_book(book_name)
	for k,v in topics.items():
		#text = ''
		chaps[k] = get_text(book, v['start'], v['end'])
		# for idx in range(v['start'], v['end']):
			# page = book.getPage(idx)
			# text = text + page.extractText()
		#chaps[k] = text
	#chaps1 = {}
	#chaps1['chap1'] = chaps['chap1']
	chapterwise_topics = {}
	chapterwise_bigrm_topics = {}
	for k,v in chaps.items():
		#text = chaps['chap1']
		print k
		text = v[0]
		sentences = get_sentences(text)
		print len(sentences)
		end = len(sentences)
		#end=8
		start = 0
		doc_clean = []
		bigrm_doc = []
		for idx in range(start,end,1):
			# print idx, sentences[idx]
			sent = sentences[idx]
			#print sent
			words = get_words_list(sent)
			#print("Raw frequency distribution")
			#print_most_common_freq_words(words)
			stop_free_words = remove_stopwords(words)
			#print("Frequency distribution after removal of stopwords")
			#print_most_common_freq_words(stop_free_words)
			punct_free_words = remove_punctuations_and_small_words(stop_free_words, 2)
			#print("Frequency distribution after removal of common words")
			#print_most_common_freq_words(punct_free_words)
			stemmed = stem_words(punct_free_words)
			#print("Frequency distribution after stemming")
			#print_most_common_freq_words(stemmed)
			lemmatized = lemmatize_words(punct_free_words)
			#print("Frequency distribution after lemmatization")
			#print_most_common_freq_words(lemmatized)

			#Bigrams
			bigrm = list(nltk.bigrams(lemmatized))
			#print("Bigrams")
			#print_most_common_freq_words(bigrm)
			bigrm_doc.append([' '.join((a,b)) for a,b in bigrm])
			#lemma_text = ' '.join(lemmatized)
			#print("POS tagging")
			pos_text = pos_tagging(lemmatized)
			#print(pos_text)
			tag_fd = [tag for (word, tag) in pos_text]
			#print_most_common_freq_words(tag_fd)
			#print("Nouns")
			noun_text = get_pos_text(pos_text, ["NN", "VBG", "VBN"])
			#print(noun_text)
			#print type(lemmatized)
			#normalized = ' '.join(lemmatized)
			doc_clean.append(lemmatized)

		print "Entering to Topic Modeling"
		print "Length of doc_clean list %s" %len(doc_clean)
		#print doc_clean
		dictionary = get_dict(doc_clean)
		doc_term_matrix = get_df_matrix(doc_clean, dictionary)
		#print "Doc term matrix"
		#print doc_term_matrix
		topic_count = 3
		ldamodel = fit_lda_model(doc_term_matrix, dictionary, topic_count)
		print("Topics unigram")
		# print(ldamodel.print_topics(num_topics=topic_count, num_words=3))
		chapterwise_topics[k] = ldamodel
		print "TFIDF model"
		tfidfmodel = fit_tfidf_model(doc_term_matrix)
		print tfidfmodel

		# Bigram topic modeling
		# print "Entering to Bigram Topic Modeling"
		# print "Length of bigrm_doc list %s" %len(bigrm_doc)
		#print bigrm_doc
		# big_dictionary = get_dict(bigrm_doc)
		# big_doc_term_matrix = get_df_matrix(bigrm_doc, big_dictionary)
		#print "Bigram Doc term matrix"
		#print big_doc_term_matrix
		# big_topic_count = 3
		# big_ldamodel = fit_lda_model(big_doc_term_matrix, big_dictionary, big_topic_count)
		# print("Bigram topics")
		# print(big_ldamodel.print_topics(num_topics=big_topic_count, num_words=3))
		# chapterwise_bigrm_topics[k] = big_ldamodel
		# print "Bigram TFIDF model"
		# big_tfidfmodel = fit_tfidf_model(big_doc_term_matrix)
		# print big_tfidfmodel
		print 50*"@"
		hdp = fit_hdp_model(doc_term_matrix, dictionary)
		print hdp.print_topics(num_topics=3, num_words=3)
		print 50*"@"

	print 50 * "*"
	print len(chapterwise_topics)
	for chap, model in chapterwise_topics.items():
		print chap, model.print_topics(num_topics=3, num_words=3)
		import matplotlib.pyplot as plt
		from wordcloud import WordCloud
		for t in range(model.num_topics):
			wc = WordCloud().fit_words(dict(model.show_topic(t, 200)))
			plt.figure()
			plt.imshow(wc, interpolation='bilinear')
			plt.axis("off")
			plt.title("Topic #" + k+str(t))
			# plt.show()
			plt.savefig(chap+str(t))
	#print len(chapterwise_bigrm_topics)
	# print 50 * "#"
	# for chap, model in chapterwise_bigrm_topics.items():
	# 	print chap, model.print_topics(num_topics=3, num_words=3)

# distance metrics are cosine distance and Hellingser distance

# https://markhneedham.com/blog/2015/02/12/pythongensim-creating-bigrams-over-how-i-met-your-mother-transcripts/
# https://estnltk.github.io/estnltk/1.1/index.html
