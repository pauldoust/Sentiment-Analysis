from pr.nlp.NlpHelper import * 
from pr.dataloading.DataHandler import * 
import gensim
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim import corpora
import pandas as pd
import re
import heapq
import random

def main():
	topics_no = 5
	datasetFrames = []
	ldamodel = loadLdaModel('lda.model', num_topics = topics_no, passes = 100)
	topics = ldamodel.print_topics(num_topics=topics_no , num_words=10) 
	for t in topics:
		topic_terms = ldamodel.get_topic_terms(t[0])
		topic_words = " ".join([ldamodel.id2word[term] for term,prob in topic_terms])
		print("Topic #" + str(t[0] + 1) + ": ", topic_words)
		# print("Topic #" + str(t[0] + 1) + ": ", t[1])

	datasetFrames = ProcessDataset("processedData.xlsx")
	columns = ['Topic ' + str(i)  for i in range(1,topics_no + 1)]
	probArray = pd.DataFrame(columns = columns)
	if FilesHandler.ifFileExists("TopicDoc.xlsx"):
		print("Probability Distribution found, loading..")
		distFile = ExcelHandler.loadFromDirectory(".", format="xlsx", isDebug=True, specificFileName="TopicDoc.xlsx" )
		distFrames = ExcelHandler.loadIntoDataframe(distFile)
		N = 500
		# distFrames.dropna(how="all", inplace=True)
		# f = list(range(0, 500))
		# f = distFrames.loc[0, 'Topic 1':'Topic 500'].columns  # retrieve only the 0th row for efficiency
		f = ['Topic ' + str(i) for i in range(1,N + 1)]
		# f = f.append()
		# f = ['Topic 1', 'Topic 2']
		distFrames.dropna(subset=f, inplace = True, how='all')
		distFrames.dropna(subset=["Applications"], inplace = True, how='all')
		# distFrames.dropna(how="all", inplace=True)
		docTopicframes = distFrames.iloc[:,0:N]

		docTopicframes.dropna(subset=f, inplace=True)
		# docTopicframes.dropna(how="all", inplace=True)
		docTopicframes = docTopicframes.values 
		# print(len(docTopicframes))
		similarity_matrix = np.array(len(docTopicframes)**1*[0],np.float).reshape(1,len(docTopicframes))
		targets = random.sample(range(0, len(docTopicframes)), 4)
		print("randomized: ", targets)
		print("----------------------------------")
		for t in targets:
			print("Target #: ", t)
			print("Target: ", distFrames.iloc[t]["Applications"])
			target =docTopicframes[t]
			target = target.reshape(1,N)
			m = 0
			target = target[0]
			# print("trag:", target)
			# print(docTopicframes[0])
			# return
			# print(leng)
			for n in range(len(docTopicframes)):
				# print(n)
				similarity_matrix [m,n] = custom_similarity(target, docTopicframes[n],0.001,N)
			max_n = 3

			# print (similarity_matrix[0])
			# print("max: ")
			# return
			maxn =np.argpartition(similarity_matrix[0], -max_n)[-max_n:]
			print( maxn)
			for s in maxn:
				print(s,distFrames.iloc[s]["Applications"])
			print("----------------------------------")
	else:
		print("Probability Distribution not found, creating..")
		f0 = {'Topic ' + str(i)  : str(0) for i in range(1,topics_no + 1)}
		for index, row in datasetFrames.iterrows():
			print("index: ",index)
			if isinstance(row["Summary"],float):
				continue
			f = f0
			for top,pro in ldamodel[ldamodel.id2word.doc2bow(row["Summary"].split())]:
				f['Topic '+ str(top+1)] = pro
			probArray.loc[index] = f		
			
		probArray["polarity"] = datasetFrames["polarity"]
		# probArray["Benefits"] = datasetFrames["Benefits"]
		# probArray["Applications Chunks"] = probArray["Applications"]
		# probArray["Benefits Chunks"] = probArray["Benefits"]
		# probArray["Applications Chunks"] = probArray["Applications Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(x))
		# probArray["Benefits Chunks"] = probArray["Benefits Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(x))
		# probArray["Applications Chunks"] = probArray["Applications Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(re.sub(r'[\*|o|•] \s+ ([A-Z]+)',r'. \1',x)))
		# probArray["Benefits Chunks"] = probArray["Benefits Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(re.sub(r'[\*|o|•] \s+ ([A-Z]+)',r'. \1',x)))
		# probArray["Applications Chunks"] = probArray["Applications Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(re.sub(r'o ([A-Z]+)',r'. \1',x),customToExclude=["*","•"]))
		# probArray["Benefits Chunks"] = probArray["Benefits Chunks"].apply(lambda x:  "" if isinstance(x, float) else PreProcessing.shallowParsing(re.sub(r'o ([A-Z]+)',r'. \1',x),customToExclude=["*","•"]))
		writer = pd.ExcelWriter("TopicDoc.xlsx")
		probArray.to_excel(writer,'Sheet1')

	docs = []
	# print('inferred')
	# print(ldamodel[ldamodel.id2word.doc2bow(doc_complete[0].split())])

	return

def loadLdaModel(modelName, num_topics = 10, passes= 1):
	if FilesHandler.ifFileExists(modelName):
		print("Loading Trained Model...")
		ldamodel =  gensim.models.LdaModel.load(modelName)
		print("Model Loaded")
		print("Extracted Latent Topics: ")
	else:
		print("Model not found, training..")
		datasetFrames = ProcessDataset("processedData.xlsx")
		doc_complete  = datasetFrames["Summary"].as_matrix()
		doc_clean = [PreProcessing.clean(doc).split() for doc in doc_complete]
		dictionary = corpora.Dictionary(doc_clean)
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		Lda = gensim.models.ldamodel.LdaModel
		ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=passes)
		# topics = ldamodel.print_topics(num_topics=50, num_words=10) 
		ldamodel.save('lda.model')
		print("Model trained and saved successfully")
	return ldamodel

def ProcessDataset(datasetName):
	if FilesHandler.ifFileExists(datasetName):
		print("Pre-Processed Dataset found, Loading..")
		dataFiles = ExcelHandler.loadFromDirectory(".", format="xlsx", isDebug=True, specificFileName=datasetName )
		datasetFrames =ExcelHandler.loadIntoDataframe(dataFiles)
	else:
		print("Pre-Processed Dataset not found, Processing..")

		dataFiles = ExcelHandler.loadFromDirectory("Dataset/", format="", isDebug=True)
		datasetFrames =ExcelHandler.loadIntoDataframe(dataFiles)

		datasetFrames = datasetFrames.replace(np.nan, '', regex=True)
		datasetFramesToProcess = datasetFrames.copy()

		# datasetFramesToProcess['Summary'] = pd.Series(datasetFramesToProcess.fillna(' ').values.tolist()).str.join(' ')
		datasetFramesToProcess['Summary'] = datasetFramesToProcess['text'].apply(lambda x:  PreProcessing.clean(x,customStopWords = ["*","•","«","»","،","،،"], lang="arabic", withStemming=True))

		# datasetFrames['Summary'] = pd.Series(datasetFrames.fillna('').values.tolist()).str.join(' ')
		# datasetFrames['Summary'] = datasetFrames['Summary'].apply(lambda x:  PreProcessing.clean(x,customStopWords = ["*","•"]))
		datasetFrames['Summary'] = datasetFramesToProcess['Summary']
		writer = pd.ExcelWriter(datasetName)
		datasetFrames.to_excel(writer,'Sheet1')
		print("Dataset is processed and saved in the same directory")

	print("Data Description: ")
	ExcelHandler.describe(datasetFrames)
	return datasetFrames




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
def kmean(mydocs, clusters_num):
	# mydocs = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The generation of random binary unordered trees", "The intersection graph of paths in trees", "Graph minors IV Widths of trees and well quasi ordering", "Graph minors A survey"]
	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(mydocs)
	true_k = clusters_num
	model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
	model.fit(X)
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(true_k):
		print ("Cluster %d: " % i)
		w=""
		for ind in order_centroids[i, :]:
			w = w + " " + terms[ind]
			# print (' ' + terms[ind])
		print(w)
	return order_centroids

def wardClustering(mydocs):
	if FilesHandler.ifFileExists("linkage_matrix.dat"):
		linkage_matrix = np.load("linkage_matrix.dat")
		print(linkage_matrix)
	else:
		print("ward clustering")
		vectorizer = TfidfVectorizer(stop_words='english')
		X = vectorizer.fit_transform(mydocs)
		# print("docs: ")
		# print(mydocs)
		# print("X")		
		# print(X)
		dist = 1 - cosine_similarity(X)
		# print("dist")
		# print(dist)
		linkage_matrix = ward(dist)
		# print("linksage")
		# print(linkage_matrix)

		linkage_matrix.dump("linkage_matrix.dat")

	# print(type(linkage_matrix))
	# print(linkage_matrix.shape)
	# print(linkage_matrix[5,:])
	fig, ax = plt.subplots()
	ax = dendrogram(linkage_matrix, orientation="right");
	plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
	plt.tight_layout()
	plt.savefig('ward_clusters.png', dpi=1024) #save figure as ward_clusters
	plt.close()
	print("Clustering Ended")
	return linkage_matrix

main()
