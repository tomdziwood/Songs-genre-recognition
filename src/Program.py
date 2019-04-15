import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report



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



def main():
	print("Poczatek programu.")
	np.random.seed(0)
	
	sciezka_danych = '../data/'
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
	
	print("Koniec programu.")

main()