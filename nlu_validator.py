import warnings
import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_data(file, test_size=0.2):
    df = pd.read_csv(file)
    print(df['intent'].value_counts())
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[['intent']])
    print(train['intent'].value_counts())
    print(test['intent'].value_counts())

    #train2 = df.sample(frac=0.8, random_state=42)
    #test2 = df.drop(train.index)
    #print(train2['intents'].value_counts())
    #print(test2['intents'].value_counts())
    return train, test

from nlu_config import DEFAULT_RASA_CONFIG
def train_rasa_nlu(train, version, config=DEFAULT_RASA_CONFIG):
    from rasa_nlu.training_data.formats.markdown import MarkdownReader
    from rasa_nlu.model import Trainer
    from rasa_nlu.config import RasaNLUModelConfig

    intents = train['intent'].unique().tolist()
    train = dict(tuple(train.groupby('intent')))
    lines = []
    for intent in intents:
        intent_phrases = train[intent]['utterance']
        lines.append('## intent:' + intent)
        for phrase in intent_phrases:
            lines.append('- ' + phrase)
    training_data = MarkdownReader().reads("\n".join(lines))
    trainer = Trainer(RasaNLUModelConfig(config))
    print('Training...')
    trainer.train(training_data)
    print('Saving model...')
    model_directory = trainer.persist('models/rasa/',
                                      project_name='benchmark',
                                      fixed_model_name=version)
    return trainer

def test_rasa_nlu(test, version):
    from rasa_core.interpreter import RasaNLUInterpreter

    print('Loading model...')
    interpreter = RasaNLUInterpreter('models/rasa/benchmark/'+version)

    test['predict_intent'] = ''
    test['match'] = 0

    print('Testing...')

    done = 0
    total = len(test)
    for index, row in test.iterrows():
        parse_data = interpreter.parse(row['utterance'])
        done += 1
        printProgress(done,total)
        intent = parse_data['intent'];
        if intent is not None:
            test.at[index, 'predict_intent'] = intent['name']
            if row['intent'] == intent['name']:
                test.at[index, 'match'] = 1
    # Percentage of correct predictions
    missed = test[test['match'] == 0]
    accuracy = 100 * (1-len(missed)/len(test))
    print('Rasa NLU scores %0.2f%% with %d false predictions in total %d samples' % (accuracy, len(missed), len(test)))
    save_csv(missed, 'missed/'+version+'/rasa.csv')
    result = test['match'].value_counts()
    test['match'].astype(int).plot.hist();
    return result;

from nlu_config import DEFAULT_SNIPS_CONFIG
def train_snips_nlu(train, version, config=DEFAULT_SNIPS_CONFIG):
    from snips_nlu import load_resources
    from de.core.snips_custom_units.de_nlu_engine import DESnipsNLUEngine

    interpreter = DESnipsNLUEngine(config)

    snips_intents = {}
    snips_entities = {}
    training_data = {'language': 'en', 'intents': snips_intents, 'entities': snips_entities}

    intents = train['intent'].unique().tolist()
    train = dict(tuple(train.groupby('intent')))

    for intent in intents:
        utterances = []
        snips_intent = {'utterances': utterances}
        snips_intents[intent] = snips_intent
        intent_phrases = train[intent]['utterance']
        for phrase in intent_phrases:
            items = []
            items.append({'text': phrase})
            utterance = {'data': items}
            utterances.append(utterance)
    load_resources("en")

    print('Training model...')
    interpreter.fit(training_data)
    model = interpreter.to_dict()

    print('Saving model...')
    save_obj(model,'models/snips/benchmark/', version)

def test_snips_nlu(test, version):
    from snips_nlu import load_resources
    from de.core.snips_custom_units.de_nlu_engine import DESnipsNLUEngine


    print('Loading model...')
    load_resources("en")

    model = load_obj('models/snips/benchmark/', version)
    interpreter = DESnipsNLUEngine.from_dict(model)

    test['predict_intent'] = ''
    test['match'] = 0

    print('Testing...')

    done = 0
    total = len(test)
    for index, row in test.iterrows():
        parse_data = interpreter.parse(row['utterance'])
        done += 1
        printProgress(done, total)
        intent = parse_data[0]['intent'];
        if intent is not None:
            test.at[index, 'predict_intent'] = intent['intentName']
            if row['intent'] == intent['intentName']:
                test.at[index, 'match'] = 1

    # Percentage of correct predictions
    missed = test[test['match'] == 0]
    accuracy = 100 * (1-len(missed)/len(test))
    print('Snips NLU scores %0.2f%% with %d false predictions in total %d samples' % (accuracy, len(missed), len(test)))
    save_csv(missed, 'missed/'+version+'/snips.csv')
    result = test['match'].value_counts()
    test['match'].astype(int).plot.hist();
    return result;

def test_tf_nlu(train, test, version):
    from tf_intent_classifier import IntentClassifier
    interpreter = IntentClassifier('elmo')
    #interpreter.train(train, version)
    interpreter.test(test, version)

if __name__ == "__main__":
    #train, test = load_data('snips.train.csv', 0.8)
    #test_tf_nlu(train, test, '2.8')
    #train, test = load_data('snips.train.csv', 0.2)
    #train_snips_nlu(train, '2.8')
    #test_snips_nlu(test,'2.8')
    #train_rasa_nlu(train, '2.8')
    #test_rasa_nlu(test,'2.8')

    train, test = load_data('smalltalk.csv', 0.2)
    #train_snips_nlu(train, 'smalltalk')
    #test_snips_nlu(test, 'smalltalk')
    #train_rasa_nlu(train, 'smalltalk')
    #test_rasa_nlu(test, 'smalltalk')
    test_tf_nlu(train, test, 'smalltalk')
    pass