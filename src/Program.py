import os
import numpy as np
import pandas as pd


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
		print("len(dataset_part) = " + str(len(dataset_part)))
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
	
	[dataset, train, test] = wczytajDane(sciezki_plikow, gatunki)
	
	print("\n\nElementow w zbiorze treningowym: {train}, testowym: {test}".format(train=len(train), test=len(test)))

	print("\n\nLicznosc klas w zbiorze treningowym: ")
	print(train.Gatunek.value_counts())  # wyswietl rozklad etykiet w kolumnie "Gatunek"

	print("\n\nLicznosc klas w zbiorze testowym: ")
	print(test.Gatunek.value_counts())   # wyswietl rozklad etykiet w kolumnie "Gatunek"
	
	print("Koniec programu.")

main()