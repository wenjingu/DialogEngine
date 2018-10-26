
DEFAULT_RASA_CONFIG = {
    "language": "en",
    "pipeline": "spacy_sklearn",
    #"pipeline": [{"name":"nlp_spacy"}, {"name":"tokenizer_spacy"}, {"name":"intent_entity_featurizer_regex"}, {"name":"intent_featurizer_spacy"},
    #             {"name":"ner_crf"}, {"name":"ner_synonyms"}, {"name":"ner_duckling"}, {"name":"intent_classifier_sklearn"}],
    "data": None,
}

DEFAULT_SNIPS_CONFIG = {
    u'intent_parsers_configs': [
        {
            u'max_entities': 200,
            u'max_queries': 50,
            u'unit_name': u'deterministic_intent_parser'
        },
        {
            u'intent_classifier_config': {
                u'data_augmentation_config': {
                    u'min_utterances': 20,
                    u'noise_factor': 5,
                    u'unknown_word_prob': 0,
                    u'unknown_words_replacement_string': None
                },
                u'featurizer_config': {u'sublinear_tf': False},
                u'random_seed': None,
                u'unit_name': u'log_reg_intent_classifier_all_cls'
            },
            u'slot_filler_config': {
                u'crf_args': {
                    u'algorithm': u'lbfgs',
                    u'c1': 0.1,
                    u'c2': 0.1
                },
                u'data_augmentation_config': {
                    u'capitalization_ratio': 0.2,
                    u'min_utterances': 200
                },
                u'exhaustive_permutations_threshold': 64,
                u'feature_factory_configs': [
                    {
                        u'args': {
                            u'common_words_gazetteer_name': u'top_10000_words',
                            u'n': 1,
                            u'use_stemming': True
                        },
                        u'factory_name': u'ngram',
                        u'offsets': [-2,
                                     -1,
                                     0,
                                     1,
                                     2]
                    },
                    {
                        u'args': {
                            u'common_words_gazetteer_name': u'top_10000_words',
                            u'n': 2,
                            u'use_stemming': True
                        },
                        u'factory_name': u'ngram',
                        u'offsets': [-2,
                                     1]
                    },
                    {
                        u'args': {},
                        u'factory_name': u'is_digit',
                        u'offsets': [-1,
                                     0,
                                     1]
                    },
                    {
                        u'args': {},
                        u'factory_name': u'is_first',
                        u'offsets': [-2,
                                     -1,
                                     0]
                    },
                    {
                        u'args': {},
                        u'factory_name': u'is_last',
                        u'offsets': [0,
                                     1,
                                     2]
                    },
                    {
                        u'args': {u'n': 1},
                        u'factory_name': u'shape_ngram',
                        u'offsets': [0]
                    },
                    {
                        u'args': {u'n': 2},
                        u'factory_name': u'shape_ngram',
                        u'offsets': [-1,
                                     0]
                    },
                    {
                        u'args': {u'n': 3},
                        u'factory_name': u'shape_ngram',
                        u'offsets': [-1]
                    },
                    {
                        u'args': {
                            u'tagging_scheme_code': 2,
                            u'use_stemming': True},
                        u'drop_out': 0.5,
                        u'factory_name': u'entity_match',
                        u'offsets': [-2,
                                     -1,
                                     0]
                    },
                    {
                        u'args': {
                            u'tagging_scheme_code': 1},
                        u'factory_name': u'builtin_entity_match',
                        u'offsets': [-2,
                                     -1,
                                     0]
                    },
                    {
                        u'args': {
                            u'cluster_name': u'brown_clusters',
                            u'use_stemming': False
                        },
                        u'factory_name': u'word_cluster',
                        u'offsets': [-2,
                                     -1,
                                     0,
                                     1]
                    }
                ],
                u'random_seed': None,
                u'tagging_scheme': 1,
                u'unit_name': u'crf_slot_filler_with_probs'
            },
            u'unit_name': u'probabilistic_intent_parser_all_cls_ext'
        }
    ],
    u'unit_name': u'de_nlu_engine'
}
