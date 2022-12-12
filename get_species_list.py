import collections
import json


if __name__ == '__main__':
    with open('data/train_annotation.json') as f:
        data = json.load(f)
    
        species_count = collections.Counter()
        for i in range(len(data['data'])):
            species_count[data['data'][i]['species']] += 1

        print(species_count, len(species_count))

    species = list(species_count.keys())
    print({specie: i for i, specie in enumerate(species)})

    with open('data/val_annotation.json') as f:
        data = json.load(f)
    
        species_count = collections.Counter()
        for i in range(len(data['data'])):
            species_count[data['data'][i]['species']] += 1

        print(species_count, len(species_count))

    
    with open('data/test_prediction.json') as f:
        data = json.load(f)
    
        species_count = collections.Counter()
        for i in range(len(data['data'])):
            species_count[data['data'][i]['species']] += 1

        print(species_count, len(species_count))