# The-eeg-pre-processing-script-for-python

Only few steps for EEG raw data batch preprocessing 

## Step 1: Update CSV file with IDs (participants) and raw file paths
update metric file with your raw data (.edf) and initial the metric with False by default: ` meta/batch_processing_metric.csv `

## Step 2: Execute eeg_processing under current path
` python ./eeg_processing.py `

## Step 3: Check logs for error detection
` logs/eeg-logs.log `

---

## Settings:

### Setting I-global configuration: .Env configuration
The default configuration will affects all the batch files.

### Setting II-customize the scripts: the functions in each .py file
like `eeg_preprocessing.py`

` def get_experimental_raw_list_from_annotations(raw, training_prase, experimental_prase)`

` def run_preprocessing(args) ` ...

