import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import json
import pickle
import urllib
from utils import *

from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer,LabelEncoder

MODULE_SPEC = {
    'universal':"https://tfhub.dev/google/universal-sentence-encoder/2",
    'elmo':"https://tfhub.dev/google/elmo/2"
}

def disable_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class IntentClassifier(object):
    def __init__(self, embedding) :
        self.embedding = embedding

    def train(self, train, version):
        #disable_gpu()
        train_utterances = train['utterance'].astype('str')
        train_intents = train['intent']

        encoder = LabelEncoder()
        encoder.fit_transform(train_intents)
        train_encoded = encoder.transform(train_intents)
        num_classes = len(encoder.classes_)

        print(encoder.classes_)

        embeddings = hub.text_embedding_column('utterance',
                                                module_spec=MODULE_SPEC[self.embedding],
                                                trainable=False)

        multi_class_head = tf.contrib.estimator.multi_class_head(
            num_classes,
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        features = {
            "utterance": np.array(train_utterances).astype(np.str)
        }
        labels = np.array(train_encoded).astype(np.int32)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True, batch_size=32,
                                                            num_epochs=25)
        #train_input_fn = tf.estimator.inputs.pandas_input_fn(
        #    train, train["intent"], num_epochs=None, shuffle=True)
        estimator = tf.contrib.estimator.DNNEstimator(
            head=multi_class_head,
            hidden_units=[64, 10],
            model_dir='models/tf/benchmark/'+self.embedding+'/'+version,
            #optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
            feature_columns=[embeddings])

        estimator.train(input_fn=train_input_fn)

    def test(self, test, version):

        disable_gpu()
        test_utterances = test['utterance'].astype('str')
        test_intents = test['intent']

        test['predict_intent'] = ''
        test['match'] = 0

        encoder = LabelEncoder()
        encoder.fit_transform(test_intents)
        test_encoded = encoder.transform(test_intents)
        num_classes = len(encoder.classes_)

        embeddings = hub.text_embedding_column('utterance',
                                                module_spec=MODULE_SPEC[self.embedding],
                                                trainable=False)

        multi_class_head = tf.contrib.estimator.multi_class_head(
            num_classes,
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        estimator = tf.contrib.estimator.DNNEstimator(
            head=multi_class_head,
            hidden_units=[64, 10],
            model_dir='models/tf/benchmark/' + self.embedding + '/' + version,
            feature_columns=[embeddings])

        predict_input_fn = tf.estimator.inputs.numpy_input_fn({"utterance": np.array(test_utterances).astype(np.str)},
                                                              shuffle=False)
        results = estimator.predict(predict_input_fn)

        index = 0
        total = len(test)
        predict_intent_idx = test.columns.get_loc('predict_intent')
        match_idx = test.columns.get_loc('match')
        # Display predictions
        for result in results:
            idx = np.argmax(result['probabilities'])
            intent = encoder.classes_[idx]
            row = test.iloc[index]
            test.iat[index, predict_intent_idx] = intent
            if row['intent'] == intent:
                test.iat[index, match_idx] = 1
            index += 1
            printProgress(index, total)

        # Percentage of correct predictions
        missed = test[test['match'] == 0]
        accuracy = 100 * (1 - len(missed) / len(test))
        print('DNN NLU scores %0.2f%% with %d false predictions in total %d samples' % (
        accuracy, len(missed), len(test)))
        save_csv(missed, 'missed/' + version + '/' + self.embedding + '.tf.csv')
        result = test['match'].value_counts()
        return result;


