import os

import pandas as pd


def get_metrics(path='data/results/metrics'):
    results = []
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        if f.endswith('.csv'):
            model, dataset = f[:-4].split('_')
            df = pd.read_csv('{}/{}'.format(path,  f))
            df['model'] = model
            df['dataset'] = dataset
            results.append(df)
    results = pd.concat(results)
    results = results.groupby(['model', 'dataset']).agg(['mean', 'sem'])
    results.columns.set_names(['metric', 'statistic'], inplace=True)
    results = results.stack(['metric', 'statistic'])
    results = results.unstack(['statistic', 'dataset', 'model']).stack('model')
    return results

def get_long_metrics(path='data/results/metrics'):
    results = []
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        if f.endswith('.csv'):
            model, dataset = f[:-4].split('_')
            df = pd.read_csv('{}/{}'.format(path,  f))
            df.columns.name = 'metric'
            df = df.stack()
            df.name = 'value'
            df = df.reset_index()
            df['model'] = model
            df['dataset'] = dataset
            results.append(df)
    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    print(get_metrics())
