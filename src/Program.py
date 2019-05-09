import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from nltk import word_tokenize



def wczytajDane(sciezki_plikow, gatunki):

	#uszykowanie tablic do przechowywania wszystkich danych
	dataset = pd.DataFrame()
	train = pd.DataFrame()
	test = pd.DataFrame()
	
	for iGatunek in range(len(gatunki)):
		#otwarcie pliku
		dataset_part = pd.read_csv(sciezki_plikow[iGatunek], sep = "\n", header = None, names = ["Tekst"])
		dataset_part["Gatunek"] = gatunki[iGatunek]
		dataset_part["Gatunek_indeks"] = iGatunek
		dataset = dataset.append(dataset_part)
		print("\nlen(dataset_part) = " + str(len(dataset_part)))
		print("dataset_part.shape = " + str(dataset_part.shape))
		print(dataset_part[:5])
		
		#wskazanie indeksow dokumentow, ktore uzyte zostana jako zbior trenujacy
		train_indices = np.random.rand(len(dataset_part)) < 0.7
		
		#wydzielenie zbioru trenujacego
		train_part = dataset_part[train_indices]
		train = train.append(train_part)
		print("len(train_part) = " + str(len(train_part)))
		print("train_part.shape = " + str(train_part.shape))
		
		#wydzielenie zbioru testujacego
		test_part = dataset_part[~train_indices]
		test = test.append(test_part)
		print("len(test_part) = " + str(len(test_part)))
		print("test_part.shape = " + str(test_part.shape))
	
	
	print("\nlen(dataset) = " + str(len(dataset)))
	print("dataset.shape = " + str(dataset.shape))
	
	print("len(train) = " + str(len(train)))
	print("train.shape = " + str(train.shape))
	
	print("len(test) = " + str(len(test)))
	print("test.shape = " + str(test.shape))
	
	return [dataset, train, test]

	
	
def bagOfWords(train, test):
	vectorizer = CountVectorizer()
	X_train_counts = vectorizer.fit_transform(train['Tekst'])
	X_test_counts = vectorizer.transform(test['Tekst'])
	print("\nRozmiar stworzonej macierzy: {x}".format(x=X_train_counts.shape))
	print("Liczba dokumentow: {x}".format(x=X_train_counts.shape[0]))
	print("Rozmiar wektora bag-of-words {x}".format(x=X_train_counts.shape[1]))
	
	
	nb = MultinomialNB()

	nb.fit(X_train_counts, train['Gatunek'])

	print("\nIle elementow testowych udalo sie poprawnie zaklasyfikowac?")
	X_test_predict = nb.predict(X_test_counts)
	accuracy = sum(X_test_predict == test['Gatunek']) / len(test)
	print(accuracy)
	print("Szczegolowy raport (per klasa)")
	print(classification_report(test['Gatunek'], X_test_predict))
	


def load_embeddings(path):
	mapping = dict()

	with open(path, 'r', encoding='utf8') as f:
		naglowek = f.readline()

		for line in f:
			line = line.strip()
			if len(line) == 0:
				continue
			splitted = line.split(" ")
			mapping[splitted[0]] = np.array(splitted[1:], dtype=float)

	return mapping
	
	
	
def documents_to_ave_embeddings(docs, embeddings):
    result = []
    for doc in docs:
        words = word_tokenize(doc.lower())
        vectors = []
        for word in words:
            try:
                vectors.append(embeddings[word])
            except:
                pass

        srodek_ciezkosci = np.mean(vectors, axis=0)
        result.append(srodek_ciezkosci)
        
    return result
	


def wordsEmbedding(train, test):
	print("Trwa wczytywanie embeddingów...")
	mapping = load_embeddings('../word2vec/nkjp-lemmas-restricted-100-skipg-hs.txt')

	print("\nTrwa uśrednianie embeddingów:")
	print("   - zbioru trenującego...")
	train_transformed = documents_to_ave_embeddings(train['Tekst'], mapping)
	print("   - zbioru testującego...")
	test_transformed = documents_to_ave_embeddings(test['Tekst'], mapping)

	print("\nTrwa klasyfikacja...")
	classifier = SVC(C=1.0)
	classifier.fit(train_transformed, train['Gatunek_indeks'])
	
	accuracy = classifier.score(test_transformed, test['Gatunek_indeks'])
	
	print("W zbiorze testowym {n}% przypadków zostało poprawnie zaklasyfikowanych!".format(n=100.*accuracy))
	print(classification_report(test['Gatunek_indeks'], classifier.predict(test_transformed)))
	
	

def main():
	print("Poczatek programu.")
	np.random.seed(0)
	
	#sciezka_danych = '../data/'
	sciezka_danych = '../data1/'
	#sciezka_danych = '../data2/'
	#sciezka_danych = '../data3/'
	
	pliki_danych = os.listdir(sciezka_danych)
	sciezki_plikow = []
	for i in range(len(pliki_danych)):
		sciezki_plikow.append(sciezka_danych + pliki_danych[i])
	print("sciezki_plikow: \n" + str(sciezki_plikow))
	
	#wydzielenie nazwy gatunku muzyki z dostepnych nazw plikow
	gatunki = []
	for i in range(len(pliki_danych)):
		gatunki.append(pliki_danych[i][:pliki_danych[i].index('-')])
	print("pliki_danych:\n" + str(pliki_danych))
	print("\ngatunki:\n" + str(gatunki))
	
	#wczytanie danych z wielu plikow do pojedynczych zmiennych
	[dataset, train, test] = wczytajDane(sciezki_plikow, gatunki)
	
	print("\n\nElementow w zbiorze treningowym: {train}, testowym: {test}".format(train=len(train), test=len(test)))

	print("\n\nLicznosc klas w zbiorze treningowym: ")
	print(train.Gatunek.value_counts())  # wyswietl rozklad etykiet w kolumnie "Gatunek"

	print("\n\nLicznosc klas w zbiorze testowym: ")
	print(test.Gatunek.value_counts())   # wyswietl rozklad etykiet w kolumnie "Gatunek"
	
	#wykonaj klasyfikacje wedlug modelu BagOfWords
	bagOfWords(train, test)
	
	#wykonaj klasyfikacje wedlug modelu Word Embedding
	wordsEmbedding(train, test)
	
	print("Koniec programu.")

main()