import stanfordnlp

config = {
	'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize_model_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt',
	'lemma_model_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd_lemmatizer.pt',
	'depparse_model_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd_parser.pt',
	'depparse_pretrain_path': '/home/angel/stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt'
}
nlp = stanfordnlp.Pipeline(**config)  # Initialize the pipeline using a configuration dict
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.")  # Run the pipeline on input text
doc.sentences[0].print_tokens()
