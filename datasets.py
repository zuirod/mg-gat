import os

from ast import literal_eval
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MultiLabelBinarizer


class Dataset(object):
    def __init__(self, data, **side_info):
        data = pd.DataFrame(data, copy=True)
        if 'is_test' not in data.columns:
            data['is_test'] = False
        if 'is_tune' not in data.columns:
            data['is_tune'] = False
        data = data[['user_id', 'item_id', 'rating', 'is_test', 'is_tune']].astype({
            'user_id': int, 'item_id': int, 'rating': float, 'is_test': bool, 'is_tune': bool,
        })
        data['is_train'] = (data.is_test == False) & (data.is_tune == False)
        
        self.train = data[data.is_train == True].sample(frac=1)
        self.tune = data[data.is_tune == True].sample(frac=1)
        self.test = data[data.is_test == True].sample(frac=1)
        self.index = {'train': 0, 'tune': 0, 'test': 0}
        
        self.n_data = len(data)
        self.min = data['rating'].min()
        self.max = data['rating'].max()
        self.range = self.max - self.min
        self.data = data
        
        self.name = side_info['name'] = side_info.get('name', 'dataset')
        self.n_user = side_info['n_user'] = side_info.get('n_user', self.data['user_id'].max() + 1)
        self.n_item = side_info['n_item'] = side_info.get('n_item', self.data['item_id'].max() + 1)
        self.shape = (self.n_user, self.n_item)
        self.side_info = side_info

    def get_batch(self, mode='train', size=None):
        dataset = getattr(self, mode)
        if size is None:
            return dataset[:]
        i = self.index[mode]
        self.index[mode] = i + size if i + size < len(dataset) else 0
        return dataset[i:(i + size)]

    def save(self, path):
        folder = '{}/{}'.format(path, self.name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.data.to_csv(folder + '/data.csv', index=False)
        
        metadata = pd.DataFrame({'name': [self.name], 'n_user': [self.n_user], 'n_item': [self.n_item]})
        metadata.to_csv(folder + '/metadata.csv', index=False)
        
        for k, v in self.side_info.items():
            if sp.issparse(v):
                sp.save_npz('{}/{}.npz'.format(folder, k), v)
            elif isinstance(v, np.ndarray):
                np.save('{}/{}.npy'.format(folder, k), v)
            elif isinstance(v, pd.DataFrame):
                v.to_csv('{}/{}.csv'.format(folder, k), index=False)

    @staticmethod
    def load(path, **kwargs):
        kwargs['name'] = os.path.basename(path)
        for f in os.listdir(path):
            if f.startswith('.'):
                continue
            f_path = '{}/{}'.format(path, f)
            if f.endswith('metadata.csv'):
                s = pd.read_csv(f_path).iloc[0]
                kwargs.update(s.to_dict())
            elif f.endswith('.npz'):
                kwargs[f[:-4]] = sp.load_npz(f_path)
            elif f.endswith('.npy'):
                kwargs[f[:-4]] = np.load(f_path)
            elif f.endswith('.csv'):
                kwargs[f[:-4]] = pd.read_csv(f_path)
            else:
                print('File not loaded: ' + f)
        return Dataset(**kwargs)


class Monti(Dataset):
    def __init__(self, path, **kwargs):
        matrix = sp.coo_matrix(self._load_matlab_file(path, 'M'))
        data = pd.DataFrame({'user_id': matrix.row, 'item_id': matrix.col, 'rating': matrix.data})
        mask_test = sp.dok_matrix(self._load_matlab_file(path, 'Otest'))
        data['is_test'] = [mask_test[i, j] for i, j in zip(matrix.row, matrix.col)]
        
        kwargs['data'] = data
        kwargs['name'] = kwargs.get('name', 'monti')
        kwargs['n_user'] = kwargs.get('n_user', 3000)
        kwargs['n_item'] = kwargs.get('n_item', 3000)
        super(Monti, self).__init__(**kwargs)

    @classmethod
    def _load_matlab_file(cls, path_file, name_field):
        # https://github.com/fmonti/mgcnn
        """
        load '.mat' files
        inputs:
            path_file, string containing the file path
            name_field, string containig the field name (default='shape')
        warning:
            '.mat' files should be saved in the '-v7.3' format
        """
        db = h5py.File(path_file, 'r')
        ds = db[name_field]
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir   = np.asarray(ds['ir'])
                jc   = np.asarray(ds['jc'])
                out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
            out = np.asarray(ds).astype(np.float32).T
        db.close()
        return out


class Douban(Monti):
    def __init__(self, path='data/raw_data/mgcnn/douban/training_test_dataset.mat', **kwargs):
        kwargs['name'] = kwargs.get('name', 'Douban')
        kwargs['user_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_users'))
        super(Douban, self).__init__(path, **kwargs)


class Flixster(Monti):
    def __init__(self, path='data/raw_data/mgcnn/flixster/training_test_dataset_10_NNs.mat', **kwargs):
        kwargs['name'] = kwargs.get('name', 'Flixster')
        kwargs['user_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_users'))
        kwargs['item_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_movies'))
        super(Flixster, self).__init__(path, **kwargs)


class Movielens(Monti):
    def __init__(self, path='data/raw_data/mgcnn/movielens/split_1.mat', **kwargs):
        kwargs['name'] = kwargs.get('name', 'Movielens')
        kwargs['user_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_users'))
        kwargs['item_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_movies'))
        kwargs['n_user'] = kwargs['user_graph'].shape[0]
        kwargs['n_item'] = kwargs['item_graph'].shape[0]
        super(Movielens, self).__init__(path, **kwargs)


class YahooMusic(Monti):
    def __init__(self, path='data/raw_data/mgcnn/yahoo_music/training_test_dataset_10_NNs.mat', **kwargs): 
        kwargs['name'] = kwargs.get('name', 'YahooMusic')
        kwargs['item_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_tracks'))
        super(YahooMusic, self).__init__(path, **kwargs)


class MovieLens100K(Movielens):
    def __init__(self, path='data/raw_data/ml-100k', **kwargs):        
        user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        user = pd.read_csv(path + '/u.user', engine='python', encoding='ISO-8859-1', sep='|', names=user_columns)
        user.age = user.age.astype(float)/user.age.max()
        user.gender = user.gender.astype('category').cat.codes
        user = pd.get_dummies(user, columns=['occupation'])
        user.drop(columns=['user_id', 'zip_code'], inplace=True)
        kwargs['user_features'] = user.astype(float)

        item_columns = [
            'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
            'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western',
        ]
        item = pd.read_csv(path + '/u.item', engine='python', encoding='ISO-8859-1', sep='|', names=item_columns)
        item.drop(columns=item_columns[:6], inplace=True)
        kwargs['item_features'] = item.astype(float)

        kwargs['name'] = 'MovieLens100K'
        super(MovieLens100K, self).__init__(**kwargs)


class Yelp2018(Dataset):
    def __init__(self, state, path='data/raw_data/yelp-2018', **kwargs):
        kwargs['name'] = state.upper().replace(' ', '')
        
        print('loading user data...')
        user = pd.read_json(path + '/user.json', lines=True)
        print('loading business data...')
        item = pd.read_json(path + '/business.json', lines=True).rename(columns={'business_id': 'item_id'})
        print('loading review data...')
        data = pd.read_json(path + '/review.json', lines=True).rename(columns={'business_id': 'item_id', 'stars': 'rating'})

        print('filtering for businesses in {}...'.format(kwargs['name']))
        item = item[item['state'].str.upper().str.replace(' ', '') == kwargs['name']]
        item.reset_index(drop=True, inplace=True)
        item_ids = dict(zip(item['item_id'], item.index))
        data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0)].reset_index(drop=True)
        user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
        user_ids = dict(zip(user['user_id'], user.index))
        data['user_id'] = data['user_id'].apply(user_ids.get)
        data['item_id'] = data['item_id'].apply(item_ids.get)

        print('splitting data by year...')
        data['date'] = pd.to_datetime(data['date'])
        data['is_test'] = data['date'].dt.year == data['date'].dt.year.max()
        data['is_tune'] = data['date'].dt.year == data['date'].dt.year.max() - 1
        kwargs['data'] = data
        
        print('building user graph...')
        row, col = [], []
        for i, friends in user['friends'].items():
            for friend in friends.split(', '):
                if friend in user_ids:
                    row.append(i)
                    col.append(user_ids[friend])
        kwargs['user_graph'] = sp.coo_matrix(([1.0]*len(row), (row, col)), shape=(len(user), len(user)))
        
        print('processing user features...')
        compliments = user[[c for c in user.columns if c.startswith('compliment')]]
        compliments.columns = ['s: '.join(c.split('_')) for c in compliments.columns]
        kwargs['user_compliments'] = (compliments - compliments.min())/(compliments.max() - compliments.min())

        votes = user[['cool', 'funny', 'useful']]
        votes.columns = ['votes: {}'.format(c) for c in votes.columns]
        kwargs['user_votes'] = (votes - votes.min())/(votes.max() - votes.min())

        profile = user[['fans']]
        user['yelping_since'] = pd.to_datetime(user['yelping_since'])
        profile['yelping_since_year'] = user['yelping_since'].dt.year
        profile['yelping_since_month'] = user['yelping_since'].dt.month
        profile['yelping_since_day'] = user['yelping_since'].dt.day
        user['elite'] = user['elite'].apply(lambda x: [] if x == 'None' else x.split(', '))
        mlb = MultiLabelBinarizer()
        elite = pd.DataFrame(mlb.fit_transform(user['elite']), columns=['elite_' + c for c in mlb.classes_])
        profile = pd.concat([profile, elite], 1)
        kwargs['user_profiles'] = (profile - profile.min())/(profile.max() - profile.min())

        kwargs['user_features'] = pd.concat([
            kwargs['user_compliments'],
            kwargs['user_votes'],
            kwargs['user_profiles'],
        ], 1)
        
        print('processing business categories...')
        categories = item['categories'].apply(lambda x: [] if x is None else x.split(', '))
        mlb = MultiLabelBinarizer()
        categories = pd.DataFrame(mlb.fit_transform(categories), columns=mlb.classes_)
        df = pd.read_json(path + '/categories.json').set_index('alias')
        df['parents'] = df['parents'].apply(sorted)
        def get_path(node, df=df):
            if df.parents[node] == []:
                return df.title[node]
            return '{}: {}'.format(get_path(df.parents[node][0], df), df.title[node])
        df['path'] = df.index.to_series().apply(get_path)
        df.set_index('title', inplace=True)
        categories.columns = ['categories: {}'.format(df.path.get(c, c)) for c in categories.columns]
        kwargs['item_categories'] = categories.astype(float)
        
        print('building business graph...')
        kwargs['item_graph'] = kneighbors_graph(kwargs['item_categories'].values, 10)
        
        print('processing business attributes...')
        attributes = pd.read_csv(path + '/attributes.csv').rename(columns={'business_id': 'item_id'})
        attributes = attributes[attributes['item_id'].isin(item_ids)]
        attributes['item_id'] = attributes['item_id'].apply(item_ids.get)
        attributes = attributes.sort_values('item_id').reset_index(drop=True)
        attributes.drop(columns=[
            'item_id',
            'attributes.RestaurantsPriceRange2.unspecified',
            'attributes.NoiseLevel.average',
            'attributes.AgesAllowed.allages',
            'attributes.Alcohol.unspecified',
            'attributes.WiFi.unspecified',
            'attributes.RestaurantsAttire.unspecified',
            'attributes.Smoking.unspecified',
            'attributes.BYOBCorkage.unspecified',
        ], inplace=True)
        attributes.drop(columns=[c for c in attributes.columns if attributes[c].nunique() == 1], inplace=True)
        attributes.columns = [': '.join(c.split('.')) for c in attributes.columns]
        kwargs['item_attributes'] = attributes.astype(float)

        locations = item[['latitude', 'longitude']]
        locations.columns = ['locations: {}'.format(c) for c in locations.columns]
        locations = locations.fillna(locations.mean())
        kwargs['item_locations'] = (locations - locations.min())/(locations.max() - locations.min())
        
        print(item['hours'])
        hours = item['hours'].apply(lambda x: {} if x is None else literal_eval(str(x)))
        hours = pd.json_normalize(hours)
        for c in hours.columns:
            hours[['hours: {}: Open'.format(c), 'hours: {}: Close'.format(c)]] = hours[c].str.split('-', expand=True)
            hours.drop(columns=c, inplace=True)
        def to_float(s):
            if s is np.nan:
                return np.nan
            h, m = s.split(':')
            return float(h)/24 + float(m)/60/24
        hours = hours.applymap(to_float)
        kwargs['item_hours'] = hours.fillna(hours.mean())
        
        print('processing business checkins...')
        checkins = pd.read_json(path + '/checkin.json', lines=True)
        times = dict(zip(checkins['business_id'], checkins['time']))
        checkins = pd.Series([times.get(item_id, {}) for item_id in item['item_id']])
        checkins = pd.json_normalize(checkins).fillna(0)
        checkins.columns = ['checkins: {}'.format(': '.join(c.split('-'))) for c in checkins.columns]
        kwargs['item_checkins'] = (checkins - checkins.min())/(checkins.max() - checkins.min())

        kwargs['item_features'] = pd.concat([
            kwargs['item_categories'],
            kwargs['item_attributes'],
            kwargs['item_locations'],
            kwargs['item_hours'],
            kwargs['item_checkins'],
        ], 1)

        locations = item[['latitude', 'longitude']]
        locations.columns = ['locations: {}'.format(c) for c in locations.columns]
        kwargs['item_coordinates'] = locations 
            
        super(Yelp2018, self).__init__(**kwargs)
        print('done!')


if __name__ == '__main__':
    dataset = MovieLens100K()
    dataset.save('data/datasets')
    dataset = Dataset.load('data/datasets/MovieLens100K')
    print(dataset.name)
    print(dataset.n_user)
    print(dataset.n_item)
    print(dataset.n_data)
    print(dataset.min)
    print(dataset.max)
    print(dataset.n_data/float(dataset.n_user*dataset.n_item))
    print(len(dataset.train))
    print(len(dataset.tune))
    print(len(dataset.test))
    print(dataset.side_info)
