# wfu-ecg-buildathon
workspace for wake forest buildathon!

### files in this repo
> - _**GUIDE_data_sync.md**_: How to sync data
> - _**GUIDE_download_mimiciv_awss3.md**_: Fast MIMIC AWS download
> - _**README.md**_: Repo overview and log
> - _**check_my_downloads.py**_: Verify your downloaded files
> - _**check_sjlife_ids.py**_: SJLIFE CSV-versus-disk ID check
> - _**cross_vendor_pairs.py**_: Clinical-watch SJLIFE pair maker
> - _**ecg_akbilgic_pairs_manifest.json**_: Frozen SJLIFE pair hashes
> - _**eligible_patients.csv**_: Patients with 2+ studies
> - _**filter_eligible_patients.py**_: Builds eligible patient lists
> - _**ingest.py**_: Raw ECGs to arrays
> - _**pair_generator.py**_: MIMIC same-patient pair maker
> - _**requirements.txt**_: Python package list
> - _**sjlife_splitter.py**_: SJLIFE train/val/test split
> - _**split_manifest.json**_: Frozen MIMIC patient split
> - _**split_manifest_sjlife.json**_: Frozen SJLIFE participant split
> - _**patient_splitter.py**_: MIMIC patient split maker
> - _**verify_splits.py**_: Runs V1–V11 checks
> - _**evaluate.py**_: AUROC scoring with strata
> - _**ecg_mimic_pairs_manifest.json**_: Frozen MIMIC pair hashes

### daily log
[9-22-26] DATA
- Attempted to download the full MIMIC-IV-ECG
- Restarted download 3x before terminating it
- Refocused on the scope in the team plan, stopped wgetting everything

[9-23-26] DATA
- Created python script to save only eligible patients (93% of records, 742,691 records) 
- Used aria2c, which allowed me to send out multiple file download requests at once
- After 6 hours, recieved enough data to begin making essential files
- _**filter_eligible_patients.py**_, what I used to make sure the data we used had 2x patients
- _**eligible_patients.csv**_, the patients that were valubale by their id + how many studies they have
- Partial download on the MIMIC files, decided to pivot to Apple Watch files
- Downloaded Akbilgic repo
- Created a .dat to hex converter, but pivoted (too much potential for loss of data)
- Looked into a .dat to hex converter, found that WFDB .dat and .hea files can be read by wfdb-python
- _**wfdb-python**_, a python library that reads the files into numeric arrays, info on it in Team Manifest
- Studied how to create ingestion pipeline script in py to take raw file and output clean array
- In between utilizing _**wfdb-python**_ or _**scipy-python**_ (priority in keeping data as lossless as possible) 

[9-24-26] DATA
- Calculated target sample size based on Team Manifest (over 10k+ and 10k- validation pairs, cap of 2/patient)
- Target sample size, 30k eligible patients, ~18GB (19GB to be safe)
- Downloaded well over sample size, ~28GB/~90GB, which will be great to work with for now
- Investigated AWS S3 as an alternative to download instead of throttled PhysioNet HTTPS
- Set up AWS, no longer super throttled, downloaded sample size in 1.5 hours
- Plan to look into whether or not the data skews, s3 sync is pulling folders in alphabetical subject_id
- _**GUIDE_download_mimiciv_awss3**_, created writeup to ensure everyone can download files quickly
- Created list of questions to ask Program Lead for pipeline, but will stick to team plan and keep lossless a priority
- Working pipeline (should be done by EOD): WFDB/NumPy > Float64 > Lead I > SciPy (resample_poly) > 250Hz sampling rate > 10s length > NumPy matrix > Model Training, will propose to other teammates
- Tested current ~28GB Mimic sample's potential skew by alphabetical subject_id download, ruled it would not be a problem and the time distribution was very wide
- Calculated that sample data contained 50k unique patients, with 32k patients with 2+ studies, perfect for Machine Learning and model training
- Need to coordinate how many will be in train (70%), validation(15%), and local test (15%)
- Updated _**filter_eligible_patients.csv**_ to allow teammates to regenerate their own eligible_records_downloaded.csv file based on the dataset that they have if not SSOT yet
- Creating code for script _**patient_splitter.py**_ to separate training, validation, and local test models

[9-25-26] DATA
- Confirmed final trustworthy patient numbers, 250k unique study_ids, N-elig 32k eligible patients
- Redesigned _**filter_eligible_patients.py**_, pushed to repo
- Wrote _**split_manifest.json**_ for additional verification, SHA-256 hashes, and check results to ensure data matching
- Confirmed final verified split: Train 22k, Val 4.8k, Test 4.8k. 
- Built _**check_my_downloads.py**_ so teammates can make sure they hold the same necessary files from canonical set, and helps them detect if any of their patients are partials
- Wrote _**GUIDE_data_sync.md**_, documented how to sync data based on _**check_my_downloads.py**_, to ensure we all have the same source of truth
- Redesigned _**ingest.py**_, reinforced the data into ML and preservation layers for experimentation on best optimization methods
- Split ECG from one file into two, metadata and participant metadata, using Paticipant ID as index as advised from team plan
- Identified data quality bugs in 307 records, will identify patient 11992999 to Clinical
- Verified ML matrix (232449, 2500 float 32, no NaN/Inf values) and all checksums
- Wrote README in the Google Drive describing all deliverables, preservation files, verification files, and dataset definition files
- Uploaded all relevant project files to Google Drive in respective folders

[9-29-26] DATA
- Recorded that SJLIFE/Apple Watch data is permitted for training
- Patched _**run_akbilgic_batch**_ in _**ingest.py**_ (4 edits, backed up old copy), fixed participant ID handling
- Ran SJLIFE ingest, 480 records (240 participants, 1 clinical + 1 watch each), 6 failures (PID_131, 210, 232)
- Found the ID mismatch, CSV IDs 131/210/232 have no files and files 209/231/376 have no CSV rows
- Excluded all six until the dataset authors confirm, did not guess a mapping
- Wrote _**sjlife_splitter.py**_, participant split 50/25/25 (120/60/60), seed 42, frozen (sha256 7f9c0c25...)
- Wrote _**cross_vendor_pairs.py**_, clinical-watch pairs with MIMIC columns, frozen (sha256 ed9379eb...)
- Info: train 240 (1:1), val 3,600 (60 pos / 3,540 neg), test 3,600 (60 pos / 3,540 neg)
- Wrote _**evaluate.py**_, AUROC + SE, participant bootstrap, strata by vendor / time gap / age, test-set guard
- Wrote _**verify_splits.py**_, V1-V11 on MIMIC + SJLIFE, ran on real data: 34 pass, 0 fail, V11 pending Modeling
- Confirmed N = 161,352 patients, Nelig = 32,457, and 232,756 eligible - 307 quarantined = 232,449 ingested
- Open, six SJLIFE IDs for dataset authors, split and pair decisions to review with team



