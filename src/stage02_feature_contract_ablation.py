from pathlib import Path
import pandas as pd, numpy as np, json, time, hashlib, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, confusion_matrix)

ROOT=Path('/root/Zaid_EGCP_UNSW_External_Validation')
PROC=ROOT/'data/processed'; MAN=ROOT/'manifests'; RES=ROOT/'results'; MODELS=ROOT/'models'; PAPER=ROOT/'paper_extension'
for p in [MAN,RES,MODELS,PAPER]: p.mkdir(parents=True,exist_ok=True)

dev=pd.read_parquet(PROC/'train_dev_grouped_v1.parquet')
val=pd.read_parquet(PROC/'validation_grouped_v1.parquet')
contract=json.load(open(MAN/'feature_contract_v1.json'))

RF_PARAMS=dict(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    max_features='sqrt',
    criterion='gini',
    bootstrap=True,
    class_weight='balanced_subsample',
    random_state=20260907,
    n_jobs=-1,
)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def metrics(y,prob,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'roc_auc':float(roc_auc_score(y,prob)),
        'average_precision':float(average_precision_score(y,prob)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
        'fpr':float(fp/(fp+tn)) if fp+tn else None,
    }

def build_pipeline(features):
    cats=[c for c in ['proto','service','state'] if c in features]
    nums=[c for c in features if c not in cats]
    transformers=[]
    if cats:
        transformers.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cats))
    if nums:
        transformers.append(('num','passthrough',nums))
    pre=ColumnTransformer(transformers=transformers,remainder='drop',sparse_threshold=1.0,verbose_feature_names_out=False)
    rf=RandomForestClassifier(**RF_PARAMS)
    return Pipeline([('pre',pre),('rf',rf)]), cats, nums

y_dev=dev['label'].to_numpy(); y_val=val['label'].to_numpy()
rows=[]; model_meta={}
for name in ['FULL_42','NO_SERVICE_41','NUMERIC_ONLY_39']:
    feats=contract['candidate_contracts'][name]
    pipe,cats,nums=build_pipeline(feats)
    t0=time.perf_counter(); pipe.fit(dev[feats],y_dev); fit_s=time.perf_counter()-t0
    t1=time.perf_counter(); prob=pipe.predict_proba(val[feats])[:,1]; pred=(prob>=0.5).astype(int); infer_s=time.perf_counter()-t1
    m=metrics(y_val,prob,pred)
    m.update({'contract':name,'feature_count_raw':len(feats),'categorical_count':len(cats),'numeric_count':len(nums),
              'fit_seconds':fit_s,'validation_batch_seconds':infer_s,'validation_us_per_row':infer_s/len(val)*1e6})
    model_path=MODELS/f'unsw_rf_{name.lower()}_seed20260907.joblib'
    joblib.dump(pipe,model_path,compress=3)
    m['model_size_bytes']=model_path.stat().st_size; m['model_sha256']=sha256(model_path)
    try:
        m['encoded_feature_count']=int(len(pipe.named_steps['pre'].get_feature_names_out()))
    except Exception:
        m['encoded_feature_count']=None
    rows.append(m)
    model_meta[name]={'model_file':model_path.name,'sha256':m['model_sha256'],'params':RF_PARAMS,'metrics':m}
    print(name, 'F1',m['f1'],'Acc',m['accuracy'],'AUC',m['roc_auc'],'FPR',m['fpr'],'fit_s',round(fit_s,2),'encoded',m['encoded_feature_count'], flush=True)

# Predeclared service-shuffle stress on FULL_42 validation only.
full_model=joblib.load(MODELS/'unsw_rf_full_42_seed20260907.joblib')
full_feats=contract['candidate_contracts']['FULL_42']
rng=np.random.default_rng(20260907)
val_shuf=val[full_feats].copy()
val_shuf['service']=rng.permutation(val_shuf['service'].to_numpy())
prob_shuf=full_model.predict_proba(val_shuf)[:,1]; pred_shuf=(prob_shuf>=0.5).astype(int)
service_shuffle=metrics(y_val,prob_shuf,pred_shuf)
service_shuffle['contract']='FULL_42_SERVICE_SHUFFLED'
rows.append(service_shuffle)
print('SERVICE_SHUFFLED','F1',service_shuffle['f1'],'Acc',service_shuffle['accuracy'],'AUC',service_shuffle['roc_auc'],'FPR',service_shuffle['fpr'], flush=True)

results=pd.DataFrame(rows)
results.to_csv(RES/'feature_contract_ablation_validation_v1.csv',index=False)

full=next(r for r in rows if r['contract']=='FULL_42')
no_service=next(r for r in rows if r['contract']=='NO_SERVICE_41')
numeric=next(r for r in rows if r['contract']=='NUMERIC_ONLY_39')
shuffle=service_shuffle
f1_loss_pp=(full['f1']-no_service['f1'])*100
shuffle_f1_drop_pp=(full['f1']-shuffle['f1'])*100
shuffle_acc_drop_pp=(full['accuracy']-shuffle['accuracy'])*100

# Apply the frozen rule exactly. "Material" is operationalized here before Test as >=1 pp F1 drop under shuffle.
material_shuffle = shuffle_f1_drop_pp >= 1.0
within_tolerance = f1_loss_pp <= 1.0
if within_tolerance and material_shuffle:
    selected='NO_SERVICE_41'
    reason='NO_SERVICE_41 retained F1 within 1.0 pp of FULL_42 and service shuffling degraded FULL_42 by >=1.0 pp F1.'
else:
    candidates={'FULL_42':full['f1'],'NO_SERVICE_41':no_service['f1'],'NUMERIC_ONLY_39':numeric['f1']}
    selected=max(candidates,key=candidates.get)
    reason='Predeclared shortcut trigger was not jointly satisfied; selected highest Validation F1 among frozen contracts.'

selection={
    'version':'feature_contract_selection_v1',
    'test_labels_accessed':False,
    'random_state':20260907,
    'random_forest_params':RF_PARAMS,
    'full_f1':full['f1'],'no_service_f1':no_service['f1'],'numeric_only_f1':numeric['f1'],
    'no_service_f1_loss_percentage_points':f1_loss_pp,
    'service_shuffle_f1':shuffle['f1'],
    'service_shuffle_f1_drop_percentage_points':shuffle_f1_drop_pp,
    'service_shuffle_accuracy_drop_percentage_points':shuffle_acc_drop_pp,
    'material_shuffle_definition':'>=1.0 percentage-point validation F1 drop',
    'within_no_service_tolerance':within_tolerance,
    'material_service_dependence':material_shuffle,
    'selected_contract':selected,
    'selection_reason':reason,
    'models':model_meta,
}
(MAN/'feature_contract_selection_v1.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
