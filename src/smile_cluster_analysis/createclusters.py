import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
from clustersearch import ClusterSearch
#---------------------

DESAMPLE_RATE = 10
DESAMPLE_FILE = '../../data/desampled_' + str(DESAMPLE_RATE) + '_openface.csv'
CONFIDENCE_THRESH = 0.90
FEATURES = ['AU06_r', 'AU12_r']
DATASETS = ['voluntary_med_stakes', 'commanded_med_stakes', 
            'commanded_low_stakes']

#---------------------

def load_desample_resave(datasets, confidence_thresh, desample_rate, features):
    """Load data, remove bad confidence frames, desample.
    
    returns:
        pd.DataFrame
    """
    usecols = features + ['confidence', 'filename']
    df_list = []
    for dataset in datasets:
        print('\nDATASET: ', dataset)
        feat_file = '../../data/' + dataset + '_openface.csv'
    
        # load data into pandas DataFrame (this make take a while)
        df = pd.read_csv(feat_file, skipinitialspace=True, usecols=usecols)
        # the following are kinda wasteful in memory, but convenient
        #df['voluntary'] = 'voluntary' in dataset
        #df['med_stakes'] = 'med_stakes' in dataset
        #df['dataset'] = dataset
            
        print('\norig data loaded')
        print('df shape:', df.shape)
        print('  # data files : ', df['filename'].nunique())
        print('  # frames     : ', df.shape[0])
        
        df = df[df['confidence'] >= CONFIDENCE_THRESH]
        print('\ndata with acceptable confidence')
        print('df shape:', df.shape)
        print('  # data files : ', df['filename'].nunique())
        print('  # frames     : ', df.shape[0])
    
        df = df.iloc[::desample_rate,:]
        print('\ndata after desampling')
        print('df shape:', df.shape)
        print('  # data files : ', df['filename'].nunique())
        print('  # frames     : ', df.shape[0])
        
        print('\n------------------------------------------')
        df_list.append(df)
    
    # combine
    df_all = pd.concat(df_list, axis=0)
    df_list = [] # to allow garbage collect
    print('df merged shape:', df_all.shape)
    print('  # data files : ', df_all['filename'].nunique())
    print('  # frames     : ', df_all.shape[0])
    print('\ncolumns: ')
    
    print(df_all.columns)
    df_all.to_csv(DESAMPLE_FILE, 
                  index=False, 
                  columns=features)
    return df_all


#-----------------------------------------------
print('starting createclusters.py.....................')    

# SET PARAMETERS

# LOAD DATA
if not os.path.isfile(DESAMPLE_FILE):  
    load_desample_resave(datasets=DATASETS, 
                         confidence_thresh=CONFIDENCE_THRESH,
                         desample_rate=DESAMPLE_RATE,
                         features=FEATURES)

# RUN CLUSTERSEARCH    
c_search = ClusterSearch(infile=DESAMPLE_FILE, 
                         outdir='../../data/clustergen', 
                         #k_range=range(2,12), 
                         k_range=[5], 
                         d_range=[2], 
                         algorithm='km')
features=['AU06_r','AU12_r']
c_search.feature_search(features)

print('createclusters.py complete......................')
