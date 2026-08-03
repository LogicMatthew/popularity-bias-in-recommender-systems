import world
import dataloader
import model
import json

if world.config['dataset'] in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/" + world.config['dataset'])

if not world.config['pretrain'] and not world.config['resume']:
    padding = 12
    print('\n' + '=' * padding + ' CONFIG ' + '=' * padding)
    correct_config = world.config.copy()
    correct_config['device'] = str(correct_config['device'])
    print(json.dumps(correct_config, indent=4))
    print("Using BPR loss.")
    print('=' * padding + ' END ' + '=' * padding + '\n')

MODELS = {
    'lgcn': model.LightGCN,
    'flgcn': model.FastLightGCN
}