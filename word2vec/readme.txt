W tym folderze powinien znale�� si� plik z utworzonymi embeddingami dla polskich s��w.
Mo�na znale�� odpowiednie pliki chocia�by pod tym adresem: 'http://dsmodels.nlp.ipipan.waw.pl/'.

Nazwa za�adowanego tutaj pliku powinna by� odpowiednio u�yta w programie korzystaj�cym ze s�ownika embedding�w. Przyk�adowo, je�li zosta� tutaj sprowadzony plik 'nkjp-lemmas-restricted-100-skipg-hs.txt', to nale�y
w programie u�y� tej nazwy we wskazywanej �cie�ce otwarcia w funkcji 'wordsEmbedding':
mapping = load_embeddings('../word2vec/nkjp-lemmas-restricted-100-skipg-hs.txt')
