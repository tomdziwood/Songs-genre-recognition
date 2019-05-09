W tym folderze powinien znaleŸæ siê plik z utworzonymi embeddingami dla polskich s³ów.
Mo¿na znaleŸæ odpowiednie pliki chocia¿by pod tym adresem: 'http://dsmodels.nlp.ipipan.waw.pl/'.

Nazwa za³adowanego tutaj pliku powinna byæ odpowiednio u¿yta w programie korzystaj¹cym ze s³ownika embeddingów. Przyk³adowo, jeœli zosta³ tutaj sprowadzony plik 'nkjp-lemmas-restricted-100-skipg-hs.txt', to nale¿y
w programie u¿yæ tej nazwy we wskazywanej œcie¿ce otwarcia w funkcji 'wordsEmbedding':
mapping = load_embeddings('../word2vec/nkjp-lemmas-restricted-100-skipg-hs.txt')
