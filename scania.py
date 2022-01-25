import numpy as np
import pandas as pd
import time as time
from sklearn.metrics import confusion_matrix
import pickle
import streamlit as st
from PIL import Image


st.set_page_config(layout='centered') # set layout wide by default

st.title('Scania APS Failure Prediction')

image = Image.open('truck.jpeg')
st.image(image)

st.markdown('    ')

st.markdown('The breaking system of a car and a truck, both works on the same principle of introducing friction to the tyres, \
	         but however cars uses a hydraulic breakes that rely on fluid while trucks uses air breaks that uses compressed air.\
             The Air pressure system in trucks provides the required compressed air to the breaks for them to be disengaged. If \
             any components of the APS has a problem, this may cause a leakage of air which inturn may result in an immediate\
             engagement of the breaks and thereby a sudden halt to the vehicle. So as the air pressure system need to be used \
             at all times for maintaining the compressed air, the proper maintanace of the APS of a trucks is atmost important. '
             , unsafe_allow_html=True)

st.markdown('Hence a machine learning model was implemented in predicting whether a problem encounterd in a truck corresponds to \
           the failre of the APS or to any other components of the vehicle ', unsafe_allow_html=True)

def final_function_1(X):
    
    mm  = pickle.load(open('mm_normalizer.pkl', 'rb'))
    KNN_imputer = pickle.load(open('imputer_knn.pkl', 'rb'))
    lof  = pickle.load(open('lof.pkl', 'rb'))
    top_15 = pd.read_csv('top_15.csv')

    clf_xgb = pickle.load(open('models/xgb.pkl', 'rb'))

    
    features_to_remove_1 = ['bq_000','br_000']
    features_to_remove_2 = ['ah_000','am_0','an_000','ao_000','aq_000','ba_001','ba_002','ba_003','ba_004','ba_005','ba_006',
                            'bb_000','bg_000','bh_000','bi_000','bj_000','bl_000','bm_000','bn_000','bo_000','bp_000','bt_000',
                            'bu_000','bv_000','by_000','cc_000','cf_000','ci_000','cn_004','cn_005','co_000','cq_000','cs_005',
                            'dc_000','dn_000','dp_000','dt_000','ed_000','ee_000','ee_001','ee_002','ee_003','ee_004']
    
    X = X.replace('na',np.nan)
    X = X.drop(features_to_remove_1,axis=1)
    
    # NORMALIZING FOLLOWED BY MISSING VALUE IMPUTATION FOLLOWED BY REMOVING HIGHLY CORRELATED FEATURES
    X_norm = pd.DataFrame(mm.transform(X),columns=X.columns)
    X_imputed = pd.DataFrame(KNN_imputer.transform(X_norm),columns=X_norm.columns)
    X_imputed = X_imputed.drop(features_to_remove_2,axis=1)
    
    # FEATURE ENGINEERING
    #1
    null_count = X.isnull().sum(axis=1)
    null_count = null_count.to_numpy()    
    X_imputed['null_count'] = null_count
    
    #2
    lof_feature= lof.predict(X_imputed)    
    X_imputed['lof'] = lof_feature
    
    #3    
    quantiles = {}
    for features in top_15['Features']:
        quantiles[features] = np.percentile(X_imputed[features],[25,75])
        
    for feature in top_15['Features']:
        feature_val = X_imputed[feature]
        new_feature_25 = feature_val - quantiles[feature][0]
        new_feature_75 = feature_val - quantiles[feature][1]            

        X_imputed[feature+'_25'] = new_feature_25
        X_imputed[feature+'_75'] = new_feature_75
    
    ## BEST CLASSIFIER - STACKING CLASSIFIER

    y_pred_final = clf_xgb.predict(X_imputed)

    
    return y_pred_final



'''-------------------------------------------------------------------------------------------------------------- '''


st.header('Performance Evaluation')

st.markdown('Upload your csv file to get the class label predictions.')
st.markdown('Note : Please make sure that the uploaded file has no uneccesary columns other than the feature columns.')
uploaded_file  = st.file_uploader('',type = ['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file,index_col=0)
    if 'class'  in list(data.columns): 
        data = data.drop(['class'],axis=1)
    #data = data.head(30)    
    st.dataframe(data.head(10))    
    predict_button = st.button('Predict')

    if predict_button:
        if data is not None:
            start = time.time()
            y_pred = final_function_1(data)
            datapoints = np.arange(1,len(data)+1)
            df_pred = pd.DataFrame()
            df_pred['Datapoint'] = datapoints
            df_pred['Prediction'] = y_pred
            st.text('Predictions :')
            st.dataframe(df_pred)
            end = time.time()
            st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
            #df_pred.to_csv('df.csv')
  


