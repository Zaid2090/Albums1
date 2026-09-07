from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path('/root/Zaid_EGCP_UNSW_External_Validation')
RAW=ROOT/'data/raw'
PROC=ROOT/'data/processed'
MAN=ROOT/'manifests'
RES=ROOT/'results'
PAPER=ROOT/'paper_extension'
for p in [PROC,MAN,RES,PAPER]: p.mkdir(parents=True,exist_ok=True)

# IMPORTANT: mirror filenames are reversed relative to published row-count semantics.
train_path=RAW/'UNSW_NB15_testing-set.csv'   # 175,341 rows -> semantic development/train source
test_path=RAW/'UNSW_NB15_training-set.csv'  # 82,332 rows -> semantic official test source
train=pd.read_csv(train_path,encoding='utf-8-sig')
test=pd.read_csv(test_path,encoding='utf-8-sig')

predictors=[c for c in train.columns if c not in ['id','attack_cat','label']]
cat_cols=['proto','service','state']
num_cols=[c for c in predictors if c not in cat_cols]

# Predictor-group hashes: preserve all identical feature vectors inside one split.
hp_tr=pd.util.hash_pandas_object(train[predictors],index=False)
hp_te=pd.util.hash_pandas_object(test[predictors],index=False)
overlap_pred=set(hp_tr.unique()).intersection(set(hp_te.unique()))

# Purge official Test rows whose predictor vector already exists in the development source.
test_purged=test.loc[~hp_te.isin(overlap_pred)].copy()

# Group-disjoint Development/Validation split from the semantic train source.
gss=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=20260907)
idx_dev,idx_val=next(gss.split(train,train['label'],groups=hp_tr.values))
dev=train.iloc[idx_dev].copy()
val=train.iloc[idx_val].copy()

hd=set(pd.util.hash_pandas_object(dev[predictors],index=False).unique())
hv=set(pd.util.hash_pandas_object(val[predictors],index=False).unique())
ht=set(pd.util.hash_pandas_object(test_purged[predictors],index=False).unique())

# Conflicts among predictor-identical groups in the pooled retrieved data.
pooled=pd.concat([train,test],ignore_index=True)
pooled['_predictor_hash']=pd.util.hash_pandas_object(pooled[predictors],index=False).values
label_conflicts=pooled.groupby('_predictor_hash')['label'].nunique()
attack_cat_conflicts=pooled.groupby('_predictor_hash')['attack_cat'].nunique()

# Save strict partitions.
partitions={
    'train_dev_grouped_v1.parquet':dev,
    'validation_grouped_v1.parquet':val,
    'test_official_purged_v1.parquet':test_purged,
}
for fname,df in partitions.items():
    df.to_parquet(PROC/fname,index=False)

def sha256_path(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

# Profiles.
cat_rows=[]
for c in cat_cols:
    trvals=set(train[c].astype(str)); tevals=set(test_purged[c].astype(str))
    rates=train.groupby(c)['label'].agg(['count','mean']).sort_values('count',ascending=False)
    cat_rows.append({
        'feature':c,
        'train_unique':len(trvals),
        'purged_test_unique':len(tevals),
        'unseen_in_purged_test':len(tevals-trvals),
        'unseen_values':json.dumps(sorted(tevals-trvals)),
        'top_train_values':json.dumps(train[c].value_counts().head(12).to_dict()),
        'top_attack_rates':json.dumps({str(k):{'count':int(v['count']),'attack_rate':float(v['mean'])} for k,v in rates.head(12).iterrows()})
    })
pd.DataFrame(cat_rows).to_csv(RES/'categorical_audit_v1.csv',index=False)

num_profile=train[num_cols].describe(percentiles=[.01,.05,.5,.95,.99]).T
num_profile['missing']=train[num_cols].isna().sum()
num_profile['inf_count']=np.isinf(train[num_cols].to_numpy(dtype=float)).sum(axis=0)
num_profile.to_csv(RES/'numeric_audit_v1.csv')

# Feature contracts frozen BEFORE model comparison.
feature_contract={
    'version':'UNSW_EGCP_feature_contract_v1',
    'dataset':'UNSW-NB15',
    'binary_target':'label',
    'metadata_only':['id','attack_cat'],
    'categorical_features':cat_cols,
    'numeric_features':num_cols,
    'candidate_contracts':{
        'FULL_42':predictors,
        'NO_SERVICE_41':[c for c in predictors if c!='service'],
        'NUMERIC_ONLY_39':num_cols,
    },
    'preprocessing':{
        'tree_models':{
            'categorical':'OneHotEncoder(handle_unknown="ignore")',
            'numeric':'passthrough',
        },
        'linear_models':{
            'categorical':'OneHotEncoder(handle_unknown="ignore")',
            'numeric':'StandardScaler',
        },
        'unexpected_missing_values':'fail audit or impute only if a later manifest explicitly records the rule',
    },
    'excluded_from_inputs':{
        'id':'row identifier only',
        'attack_cat':'evaluation/reporting metadata only',
        'label':'binary target only',
    },
    'shortcut_ablation_plan':{
        'candidate':'service',
        'reason':'service categories show strongly heterogeneous attack rates and can encode service-context shortcuts',
        'validation_only_tests':['FULL_42','NO_SERVICE_41','FULL_42 with service shuffled at validation'],
        'predeclared_selection_rule':'Prefer NO_SERVICE_41 if validation F1 is within 1.0 percentage point of FULL_42 and shuffling service materially degrades FULL_42. Otherwise retain/report the better-justified contract without touching Test labels.',
    },
    'port_note':'These CSVs do not expose literal raw source/destination port identity. Derived count features such as ct_src_dport_ltm and ct_dst_sport_ltm are retained because they summarize connection behavior rather than a raw port identifier.',
}
(MAN/'feature_contract_v1.json').write_text(json.dumps(feature_contract,indent=2),encoding='utf-8')

split_manifest={
    'version':'strict_split_manifest_v1',
    'random_state':20260907,
    'source_mapping':{
        'semantic_development_file':train_path.name,
        'semantic_development_rows':len(train),
        'semantic_official_test_file':test_path.name,
        'semantic_official_test_rows':len(test),
        'mapping_basis':'published UNSW-NB15 partition row counts: 175,341 train and 82,332 test; mirror filenames are reversed',
    },
    'duplicate_control':{
        'group_key':'hash of all 42 predictor values; excludes id, attack_cat, label',
        'original_cross_split_predictor_groups':len(overlap_pred),
        'original_development_rows_in_overlap':int(hp_tr.isin(overlap_pred).sum()),
        'official_test_rows_removed_by_purge':int(hp_te.isin(overlap_pred).sum()),
        'purged_official_test_rows':len(test_purged),
        'development_validation_split':'GroupShuffleSplit(test_size=0.20, random_state=20260907)',
        'overlap_dev_validation':len(hd & hv),
        'overlap_dev_test':len(hd & ht),
        'overlap_validation_test':len(hv & ht),
    },
    'rows':{'train_dev':len(dev),'validation':len(val),'test_purged':len(test_purged)},
    'class_counts':{
        'train_dev':{str(k):int(v) for k,v in dev['label'].value_counts().sort_index().items()},
        'validation':{str(k):int(v) for k,v in val['label'].value_counts().sort_index().items()},
        'test_purged':{str(k):int(v) for k,v in test_purged['label'].value_counts().sort_index().items()},
    },
    'attack_categories':{
        'train_dev':{str(k):int(v) for k,v in dev['attack_cat'].value_counts().items()},
        'validation':{str(k):int(v) for k,v in val['attack_cat'].value_counts().items()},
        'test_purged':{str(k):int(v) for k,v in test_purged['attack_cat'].value_counts().items()},
    },
    'files':{},
}
for fname in partitions:
    p=PROC/fname
    split_manifest['files'][fname]={'sha256':sha256_path(p),'size_bytes':p.stat().st_size}
(MAN/'strict_split_manifest_v1.json').write_text(json.dumps(split_manifest,indent=2),encoding='utf-8')

audit={
    'dataset':'UNSW-NB15',
    'retrieved_shapes':{'semantic_development':[train.shape[0],train.shape[1]],'semantic_official_test':[test.shape[0],test.shape[1]]},
    'predictors':{'total':len(predictors),'categorical':cat_cols,'numeric_count':len(num_cols)},
    'missing_cells':{'development':int(train.isna().sum().sum()),'official_test':int(test.isna().sum().sum())},
    'duplicates_excluding_id':{
        'development':int(train.duplicated(subset=[c for c in train.columns if c!='id']).sum()),
        'official_test':int(test.duplicated(subset=[c for c in test.columns if c!='id']).sum()),
    },
    'predictor_identical_conflicting_binary_label_groups':int((label_conflicts>1).sum()),
    'predictor_identical_multi_attack_category_groups':int((attack_cat_conflicts>1).sum()),
    'cross_official_split_predictor_overlap_groups':len(overlap_pred),
    'official_test_rows_purged_for_predictor_overlap':int(hp_te.isin(overlap_pred).sum()),
    'strict_final_rows':{'train_dev':len(dev),'validation':len(val),'test_purged':len(test_purged)},
    'strict_group_overlap':{'dev_val':len(hd&hv),'dev_test':len(hd&ht),'val_test':len(hv&ht)},
    'categorical_unseen_in_purged_test':{c:sorted(set(test_purged[c].astype(str))-set(train[c].astype(str))) for c in cat_cols},
}
(MAN/'data_audit_summary_v1.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

report=f'''# UNSW-NB15 Data Audit and Feature Contract — Stage 01

## Retrieved data
- Semantic development source: **{len(train):,} rows × {train.shape[1]} columns**.
- Semantic official test source: **{len(test):,} rows × {test.shape[1]} columns**.
- Mirror filenames are reversed relative to the published row-count semantics; the mapping is explicitly recorded in the manifest.

## Leakage / duplicate audit
- Exact duplicate rows excluding `id`: **{int(train.duplicated(subset=[c for c in train.columns if c!='id']).sum()):,}** in development and **{int(test.duplicated(subset=[c for c in test.columns if c!='id']).sum()):,}** in official Test.
- Predictor-identical groups crossing the original train/Test boundary: **{len(overlap_pred):,}**.
- Official-Test rows removed by strict predictor-overlap purge: **{int(hp_te.isin(overlap_pred).sum()):,}**.
- Purged official Test retained: **{len(test_purged):,} rows**.
- Predictor-group overlaps after strict splitting: Dev↔Val = **{len(hd&hv)}**, Dev↔Test = **{len(hd&ht)}**, Val↔Test = **{len(hv&ht)}**.

## Stage-01 frozen partitions
- Train-development: **{len(dev):,}** rows.
- Validation: **{len(val):,}** rows.
- Purged official Test: **{len(test_purged):,}** rows.

## Feature contract
- Candidate predictors: **42**.
- Categorical: `proto`, `service`, `state`.
- Numeric: **39**.
- Excluded from model inputs: `id`, `attack_cat`, `label`.
- Frozen candidate contracts for Validation-only comparison: `FULL_42`, `NO_SERVICE_41`, and `NUMERIC_ONLY_39`.

## Shortcut-control decision
`service` is a predeclared shortcut candidate and will be tested by retraining without service and by service shuffling on Validation. Test labels will not be used for this decision.

## Scope boundary
This is an independent publication-extension study. It does **not** alter the submitted master's-thesis results.
'''
(PAPER/'STAGE_01_DATA_AUDIT.md').write_text(report,encoding='utf-8')

print(report)
print('Saved:')
for p in [MAN/'data_audit_summary_v1.json',MAN/'strict_split_manifest_v1.json',MAN/'feature_contract_v1.json',RES/'categorical_audit_v1.csv',RES/'numeric_audit_v1.csv']:
    print(' -',p)
