import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import glob, shutil, datetime, math, re, os
from tqdm import tnrange, tqdm_notebook
from multiprocessing import Process
from MyClf import MyClf

def iter_fit(model_set, X_train, y_train, save_folder='SavedModels', multithread=False, batch_size=16, train_params=None):

    if isinstance(model_set, dict):
        model_set = [(k, v) for k, v in model_set.items()]
    elif isinstance(model_set, list):	
        model_set = [(str(i), v) for i, v in enumerate(model_set)]
    else:
        model_set = [('Model', model_set)]
        
    if multithread:
        print('Queuing multithread jobs.')
        try:
            batch_start = 0
            for _ in tqdm_notebook(range(math.ceil(len(model_set)/batch_size)), desc='Threading'):
                p_list = []
                for model_name, model in model_set[batch_start: min(batch_start + batch_size, len(model_set))]:
                    clf = MyClf(model, name=model_name)
                    p = Process(target=clf.fit, args=(X_train, y_train, True, save_folder, train_params))
                    p_list.append(p)
                [p.start() for p in p_list]
                [p.join() for p in p_list]
                batch_start += batch_size

        except Exception as e:
            print('failed to multithread:', e)
            print('fitting manually')
            [MyClf(model, model_name).fit(X_train, y_train, save_file=True, save_folder=save_folder,
                                          train_params=train_params) for model_name, model in tqdm_notebook(model_set, desc='Training')]
    else:
        [MyClf(model, model_name).fit(X_train, y_train, save_file=True, save_folder=save_folder,
                                      train_params=train_params) for model_name, model in tqdm_notebook(model_set, desc='Training')]

def iter_test(X_test, y_test, y_bench, folder, params_filter):
    mc_df = pd.DataFrame()
    sk_zoo = glob.glob(folder + '/**/*.pkl', recursive=True)
    days = 1
    for k, v in params_filter.items():
        sk_zoo = [item for item in sk_zoo if re.search('_{}-{}[_.]'.format(k, v), item)]

    for ml_algo in tqdm_notebook(sk_zoo, desc='Results'):
        clf = joblib.load(ml_algo)
        model_name = clf.name
        days = clf.train_params['days']
        
        predictions = clf.predict(X_test)
        returns = predictions * y_bench
        if clf.clf_type() == 'MLPClassifier':
            mc_df.loc["Layers", model_name] = '{}'.format(clf.clf.hidden_layer_sizes)
            mc_df.loc["Activation", model_name] = clf.clf.activation
            mc_df.loc["Learning rate", model_name] = clf.clf.learning_rate_init
            mc_df.loc["Alpha", model_name] = clf.clf.alpha
            mc_df.loc["Tolerance", model_name] = clf.clf.tol
        mc_df.loc["Threshold", model_name] = clf.train_params['threshold']
        mc_df.loc["Train time", model_name] = '{:.2f}s'.format(clf.train_time)
        mc_df.loc["Total Return", model_name] = np.product(returns + 1) - 1
        mc_df.loc['CAGR', model_name] = (np.product(returns + 1))**(252/(len(returns)*days))-1
        mc_df.loc['Score', model_name] = clf.score(X_test, y_test)
        mc_df.loc['Precision: -1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[0][0]
        mc_df.loc['Recall: -1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[1][0]
        mc_df.loc['F-Score: -1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[2][0]
        mc_df.loc['Support: -1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[3][0]
        mc_df.loc['Precision: 1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[0][-1]
        mc_df.loc['Recall: 1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[1][-1]
        mc_df.loc['F-Score: 1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[2][-1]
        mc_df.loc['Support: 1', model_name] = metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), warn_for=())[3][-1]
        mc_df.loc['F-Score', model_name] = metrics.f1_score(y_test, clf.predict(X_test), average='weighted')
        mc_df.loc['Max Drawdown', model_name] = returns.min()
        mc_df.loc['Best 5-days', model_name] = np.max(pd.Series(returns).rolling(5).apply(lambda x: np.prod(x+1)-1))
        mc_df.loc['Worst 5-days', model_name] = np.min(pd.Series(returns).rolling(5).apply(lambda x: np.prod(x+1)-1))
    
    returns = -y_bench
    mc_df.loc['Total Return', 'Control'] = np.product(returns + 1) - 1
    mc_df.loc['CAGR', 'Control'] = (np.product(returns + 1))**(252/(len(returns)*days))-1
    mc_df.loc['Max Drawdown', 'Control'] = returns.min()
    mc_df.loc['Best 5-days', 'Control'] = np.max(pd.Series(returns).rolling(5).apply(lambda x: np.prod(x+1)-1))
    mc_df.loc['Worst 5-days', 'Control'] = np.min(pd.Series(returns).rolling(5).apply(lambda x: np.prod(x+1)-1))
    mc_df.sort_values('CAGR', axis=1, ascending=False, inplace=True)
    pct_columns = ['Threshold', 'Return', 'CAGR', 'Score', 'Precision', 'Recall', 'F-Score', 'Drawdown', 'Best', 'Worst']
    rows_format = [item for item in mc_df.index if any(col in item for col in pct_columns)]
    mc_df.ix[rows_format, :] = mc_df.loc[rows_format, :].applymap(lambda x: '{:.1f}%'.format(x*100) if not np.isnan(x) else x)
    return mc_df

def iter_predict(X_now, folder, params_filter, pred_hist=100):
    df_pred = pd.DataFrame()
    sk_zoo = glob.glob(folder + '/**/*.pkl', recursive=True)
    days = 1
    for k, v in params_filter.items():
        sk_zoo = [item for item in sk_zoo if re.search('_{}-{}[_.]'.format(k, v), item)]

    for ml_algo in tqdm_notebook(sk_zoo, desc='Results'):
        clf = joblib.load(ml_algo)
        model_name = clf.name
        days = clf.train_params['days']
        
        df_pred.loc['Prediction', model_name] = clf.predict(X_now)[0]
        if hasattr(clf.clf, 'predict_proba'):
            df_pred.loc['Prediction', model_name] = clf.predict_proba(X_now)[0][0]
        elif hasattr(clf.clf, 'predict_log_proba'):
            df_pred.loc['Prediction', model_name] = clf.predict_log_proba(X_now)[0][0]
        df_pred.loc['', model_name] = ''

        # df_pred.loc["Threshold", model_name] = clf.train_params['threshold']
        if clf.clf_type() == 'MLPClassifier':
            df_pred.loc["Layers", model_name] = '{}'.format(clf.clf.hidden_layer_sizes)
            df_pred.loc["Activation", model_name] = clf.clf.activation
            df_pred.loc["Learning rate", model_name] = clf.clf.learning_rate_init
            df_pred.loc["Alpha", model_name] = clf.clf.alpha
            df_pred.loc["Tolerance", model_name] = clf.clf.tol
    # df_pred.loc['Threshold', :] = df_pred.loc['Threshold', :].apply(lambda x: '{:.2f}%'.format(x*100))
    df_pred.loc['Prediction', :] = df_pred.loc['Prediction', :].apply(lambda x: '{:.2f}%'.format(x*100) if (np.abs(x) != 1) else x)

    try:
        prior_predictions = pd.read_pickle(os.path.join(os.getcwd(), folder, 'prior_predictions.pkl'))
        prior_predictions.sort_index(ascending=False, inplace=True)
        df_pred = df_pred.append(prior_predictions.head(pred_hist))
        prior_predictions.loc['Pred at ' + datetime.datetime.now().strftime("%H:%M:%S")] = df_pred.loc['Prediction'].to_dict()
    
    except:
        prior_predictions = pd.DataFrame(df_pred.loc['Prediction'].to_dict(), index=['Pred at ' + datetime.datetime.now().strftime("%H:%M:%S")])
    prior_predictions.to_pickle(os.path.join(os.getcwd(), folder, 'prior_predictions.pkl'))
   
    return df_pred

if __name__ == "__main__":
    pass
