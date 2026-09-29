# Dataset 

The project runs without a real dataset because the repository contains only synthetic smoke-test data.

## Recommended sources

### 1. NIH ChestX-ray14
Good for the computer-vision component of the project.
Use it for multi-label chest X-ray experiments, then connect the vision result to the agent.

Official NIH access:
https://nihcc.app.box.com/v/ChestXray-NIHCC

### 2. The Cancer Imaging Archive (TCIA)
Useful for CT/MRI/pathology/cancer imaging. TCIA provides de-identified cancer images and multiple collections.

Official:
https://www.cancerimagingarchive.net/

TCIA access policies apply. The archive states that users must agree to its data usage policies and restrictions.

### 3. MIMIC-IV
Useful for structured clinical/EHR information and multimodal research. Access to the main MIMIC-IV files is credentialed and requires the applicable agreement/training.

PhysioNet:
https://physionet.org/content/mimiciv/3.1/

### 4. MIMIC-IV Demo
A small public demo is available for pipeline development and does not require the full MIMIC-IV dataset.

https://www.physionet.org/content/mimic-iv-demo/2.2/

## 

Use:
- ChestX-ray14 or another permitted chest X-ray dataset for vision.
- MIMIC-IV/MIMIC-IV Demo for structured-data experimentation where permitted.
- A local evidence JSON/SQLite collection for retrieval.

## Important
Do not redistribute downloaded datasets in this project ZIP. Follow each dataset's license, terms, access controls and citation requirements.
