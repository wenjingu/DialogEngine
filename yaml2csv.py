from de2 import load_from_file
import pandas as pd

def yaml2csv(domain_path):
    columns = ['utterance', 'intent']
    df = pd.DataFrame(columns=columns)
    domain = load_from_file(domain_path)
    intents = domain['intent']
    rows = []
    for intent in intents:
        for phrase in intent.phrases:
            rows.append({'utterance':phrase, 'intent':intent['object_name']})

    df = df.append(rows, ignore_index=True)
    df.to_csv(domain_path.replace('yml','csv'), index=False)


if __name__ == '__main__':
    yaml2csv('domains/smalltalk.yml')
