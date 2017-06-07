from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc


class MLOperator(object):
    def __init__(self):
        pass

    def set_data_schema(self, data_schema):
        self.data_schema = data_schema

    def get_classifier(self, classifier, **kwargs):
        clf_name = classifier.lower()
        if clf_name == 'rf':
            return ensemble.RandomForestClassifier(**kwargs)
        elif clf_name == 'gb':
            return ensemble.GradientBoostingClassifier(**kwargs)
        elif clf_name == 'linearsvc':
            return svm.LinearSVC(**kwargs)
        elif clf_name == 'svc':
            return svm.SVC(**kwargs)
        elif clf_name == 'gaussiannb':
            return naive_bayes.GaussianNB(**kwargs)
        elif clf_name == 'xgb':
            return xgb.XGBClassifier(**kwargs)
        elif clf_name == 'logit':
            return linear_model.LogisticRegression(**kwargs)
        
    def get_samples_index(self, y, sampling):
        if type(sampling) == dict:
            sample_sizes = sampling
        elif type(sampling) == str and sampling == 'min':
            sample_sizes = dict([(label, y.value_counts().min()) \
                                 for label in y.unique()])
        elif type(sampling) == str and sampling in list(y.value_counts().index.astype(str)):
            sample_sizes = dict([(label, \
                                  y.value_counts()[int(sampling)])
                                 for label in y.unique()])
        elif type(sampling) == str and 'multi:' in sampling:
            sample_sizes = dict([(label, 
                int(sampling.replace('multi:', '')) * y.value_counts().min())
                    for label in y.unique()])
        elif type(sampling) == int:
            sample_sizes = dict([(label, sampling) for label in y.unique()])
        else:
            print 'unknown sampling method or no sampling'
            return y.index

        select_index_all = []
        for label in y.unique():
            label_index = y[y==label].index
            sample_size = sample_sizes[label]
        
            if sample_size <= len(label_index):
                replace = False
            else:
                replace = True
            select_index = np.random.choice(label_index, sample_size, \
                                            replace=replace)
            select_index_all = select_index_all + list(select_index)
    
        return select_index_all
    
    def train(self, x_train, y_train, classifier='rf', **kwargs):
        '''
        if not sampling is None:
                idx = self.get_samples_index(y_train, sampling)
        else:
            idx = x_train.index
        '''
        
        y_sampled = y_train
        X_sampled = x_train

        clf = self.get_classifier(classifier=classifier, **kwargs)
        print clf
        print 'fitting.....'
        clf.fit(X_sampled, y_sampled)

        return clf

    def test(self, clf, x_test, y_test):
        
        print 'predicting.....'
        y_pred = pd.Series(clf.predict(x_test), index=x_test.index)
        y_pred.name = 'predict'
        
        y_prob = pd.DataFrame(clf.predict_proba(x_test), index=x_test.index)
        y_prob.columns = [int(c) for c in clf.classes_]
        pred_df = pd.concat([y_test, y_pred, y_prob], axis=1)

        return pred_df    
    
    def get_pred_dfs(self, x_train, x_test, y_train, y_test, 
            classifier='rf', **kwargs):

        if classifier == 'tensorflow':
            clf = kwargs['model']
            prediction = clf.predict(x_train)
            pred_df_train = pd.DataFrame(prediction, columns=kwargs['columns'])
            pred_df_train['predict'] = pred_df_train.idxmax(axis=1)
            pred_df_train['true'] = pd.DataFrame(y_train, 
                columns=kwargs['columns']).idxmax(axis=1)

            prediction = clf.predict(x_test)
            pred_df_test = pd.DataFrame(prediction, columns=kwargs['columns'])
            pred_df_test['predict'] = pred_df_test.idxmax(axis=1)
            pred_df_test['true'] = pd.DataFrame(y_test, 
                columns=kwargs['columns']).idxmax(axis=1)

        else:
            clf = self.train(x_train, y_train, classifier=classifier, **kwargs)
            
            pred_df_train = self.test(clf, x_train, y_train)
            pred_df_test = self.test(clf, x_test, y_test)

        return pred_df_train, pred_df_test, clf

    def get_feature_importances(self, clf_type, X_train=None, y_train=None,
            clf=None, X_labels=None):

        if clf_type != 'pre-trained':
            clf = self.get_classifier(clf_type)
            clf.fit(X_train, y_train)
            X_labels = X_train.columns

        importances = clf.feature_importances_
        std = np.std([estimator_.feature_importances_ 
            for estimator_ in clf.estimators_], axis=0)

        return {'importances': importances, 'std': std, 'X_labels': X_labels}

class MLEvaluator(object):
    def __init__(self):
        pass
    
    def set_pred_df(self, pred_df):
        self.pred_df = pred_df
    
    def generate_confusion_matrix(self):
        idx = list('actual_' + self.pred_df.columns[2:].astype(str))
        cols = list('predict_' + self.pred_df.columns[2:].astype(str))
        
        conf_mx = pd.DataFrame(confusion_matrix(self.pred_df.true, 
            self.pred_df.predict), index=idx, columns=cols)
        
        return conf_mx

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        fig = plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm_text = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            cm_text = cm
            print('Confusion matrix, without normalization')

        print cm_text

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm_text[i, j], 4),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout(pad=4)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        return fig

    def plot_feature_importance(self, importances_dict, 
        num_features_imp_plt=50, **kwargs):
        
        date = importances_dict['date']
        print 'start getting features importance plot for: ', date
        importances = importances_dict['importances']
        std = importances_dict['std']
        X_labels = importances_dict['X_labels']        

        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize = (25,14))
    
        plt.title("Feature importances (top {0}) {1} {2} features".format(
            num_features_imp_plt, date, kwargs['feature_type']))
    
        plt.barh(range(num_features_imp_plt), 
            importances[indices][:num_features_imp_plt],
                color="b", yerr=std[indices][:num_features_imp_plt], 
                    align="center")    
        
        plt.yticks(range(num_features_imp_plt), 
            [X_labels[i] for i in indices[:num_features_imp_plt]])
        
        plt.ylim([-1, num_features_imp_plt])

        return fig

    def plot_feature_importance_xgb(self, clf):
        fig,ax = plt.subplots(figsize = (25, 14))
        
        ax = xgb.plot_importance(clf, ax = ax,
                                 title='Feature importance',
                                 xlabel='Importance', ylabel='Features',
                                 importance_type='gain')
        return fig

    def plot_roc(self):
        fig = plt.figure(figsize=(12,9))
        for lab in self.pred_df.true.unique():
            fpr, tpr, thresholds = roc_curve(self.pred_df.true, self.pred_df[str(lab)], pos_label=lab)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr,lw=2, label='Label {} (area = %0.2f)'.format(lab) % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = 20)
        plt.ylabel('True Positive Rate', fontsize = 20)
        plt.legend(loc="lower right", fontsize = 20)
        plt.grid()

        return fig

    def plot_learning_curve(self, df, label, variable_name):
        
        train_f1 = df[df.index=='train']
        test_f1 = df[df.index=='test']

        variable = sorted(list(train_f1[variable_name].unique()))
        train_m = train_f1.groupby(variable_name)[label].mean()
        train_std = train_f1.groupby(variable_name)[label].std()
        
        test_m = test_f1.groupby(variable_name)[label].mean()
        test_std = test_f1.groupby(variable_name)[label].std()
        
        fig = plt.figure(figsize=(12,9))
        
        plt.fill_between(variable, train_m - train_std, train_m + train_std, 
            alpha=0.1, color="r")
        
        plt.fill_between(variable, test_m - test_std, test_m + test_std, 
            alpha=0.1, color="g")
        
        train_line = plt.plot(variable, train_m, 
            'o-', color="r", label="Training score", markersize=10)
        
        test_line = plt.plot(variable, test_m, 'o-', color="g",
                 label="Testing score", markersize=10)
        
        # plt.ylim(min(test_f1[label].min(), train_f1[label].min())*0.95, 1.05)
        plt.ylim(0, 1)        
        plt.xlim(train_f1[variable_name].min()*0.8, train_f1[variable_name].max() * 1.1)
        plt.xticks(train_f1[variable_name].unique(), train_f1[variable_name].unique())
        plt.tick_params(axis='both', labelsize=15)
        plt.title('learning curve ({} label)'.format(label), fontsize=20)
        plt.xlabel(variable_name, fontsize=20)
        plt.ylabel('f1 score', fontsize=20)
        plt.legend(loc="best", fontsize=20)
        plt.grid()
            