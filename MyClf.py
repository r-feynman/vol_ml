import time
import os

class MyClf(object):
    
    train_params = None
    
    def __init__(self, clf, name=None):
        self.clf = clf
        self.name = name

    def clf_type(self):
        return type(self.clf).__name__

    def fit(self, X_train, y_train, save_file=False, save_folder='SavedModels', train_params=None):
        start_time = time.time()
        self.clf.fit(X_train, y_train)
        end_time = time.time()
        self.train_time = end_time-start_time
        self.train_score = self.clf.score(X_train, y_train)
        self.train_params = train_params
        if save_file:
            self._save(save_folder)

    def set_clf(self, clf):
        self.clf = clf

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight=None)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def _save(self, save_folder):
        from sklearn.externals import joblib
        clf_dir, clf_name = self.get_path(save_folder, return_tuple=True)
        
        if not os.path.exists(clf_dir):
            os.makedirs(clf_dir)
        joblib.dump(self, clf_dir + clf_name)

    def exists(self, train_params=None):
        try:
            exists = os.path.exists(self.get_path(train_params=train_params))
            return exists
        except:
            return False
        
    def get_path(self, save_folder='SavedModels', return_tuple=False, train_params=None):
        if not train_params:
            train_params = self.train_params
        
        days = train_params['days']
        threshold = train_params['threshold']
        from_date = train_params['from_date']
        to_date = train_params['to_date']
        rand_seed = train_params['rand_seed']
        test_size = train_params['test_size']

        if self.clf_type() == 'MLPClassifier':
            layers = self.clf.hidden_layer_sizes
            solver = self.clf.solver
            activation = self.clf.activation
            learning_rate = self.clf.learning_rate_init
            alpha = self.clf.alpha
            tolerance = self.clf.tol
            randst = self.clf.random_state
            clf_dir = '{}/{}/'.format(save_folder, self.clf_type())
            clf_name = '{}_days-{}_threshold-{}_layers-{}_solver-{}_activation-{}_lr-{}_alpha-{}_tol-{}_randst-{}.pkl'.format(self.clf_type(), days, threshold, layers, solver, activation, learning_rate, alpha, tolerance, randst)
            if self.name:
                clf_name = clf_name.replace('.pkl', '_name-{}.pkl'.format(self.name))
        else:
            clf_dir = '{}/{}/'.format(save_folder, self.clf_type())
            clf_name = '{}_days-{}_threshold-{}.pkl'.format(self.clf_type(), days, threshold)
            if self.name:
                clf_name = clf_name.replace('.pkl', '_name-{}.pkl'.format(self.name))

        if return_tuple:
            return clf_dir, clf_name
        else:
            return clf_dir + clf_name

    def get_params(self):
    	return clf.get_params()
