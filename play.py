import stanfordnlp

# You only download languages once
# Each language requires more that 1GB of disk space
# It takes time... have a coffee!

# stanfordnlp.download('en')
# stanfordnlp.download('es')
# stanfordnlp.download('fr')

# English is the default language, so you just invoke stanfordnlp.Pipeline()
# For Spanish you would call stanfordnlp.Pipeline(lang="es", treebank="es_ancora")
# fr_nlp = stanfordnlp.Pipeline(lang="fr", treebank="fr_gsd")  # This sets up a neural pipeline in French
en_nlp = stanfordnlp.Pipeline()

# a document is made of sentences
#doc = fr_nlp("Si ce discours semble trop long pour être lu en une fois, on le pourra distinguer en six parties")
# we pick our first and only sentence
#only_sentence = doc.sentences[0]

# only_sentence.print_dependencies()
# only_sentence.print_tokens()
# only_sentence.print_words()

# a sentence is made of words. Each word is tagged witha part of speech (POS)
#print(" ". join(["{} ({})".format(word.text, word.upos) for word in only_sentence.words]))

doc = en_nlp("He took up his brush and went tranquilly to work")   # to err is human")
only_sentence = doc.sentences[0]
print(" ". join(["{} ({} - {})".format(word.text, word.upos, word.feats) for word in only_sentence.words]))

doc = en_nlp("Our young researcher found a solution")   # to err is human")
only_sentence = doc.sentences[0]
print(" ". join(["{} ({} - {})".format(word.text, word.upos, word.feats) for word in only_sentence.words]))
