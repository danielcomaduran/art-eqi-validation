# ART EQI P300 Validation

## Description
This repo contains all the scripts, and functions to run optimization approach submitted to the 2023 IEEE EMBC conference. For this project we are using the EEG Quality Index (EQI) [1] as the objective function to optimize the hyper-parameters of the Artifact Removal Tool (ART) [2].

The data used for this project can be downloaded from the Temple Artifact Dataset [3].

## Instructions
To run the project, please install the Python environment using the following conda command:

```
conda env create -f art-eqi.yml
```



## References
1. S. D. Fickling, C. C. Liu, R. C. N. D’Arcy, S. Ghosh Hajra, and X. Song, “Good data? the EEG quality index for automated assessment of signal quality,” in 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), pp. 0219–0229, 2019. ISSN: 2644-3163.

2. A. K. Maddirala and K. C. Veluvolu, “Eye-blink artifact removal from single channel EEG with k-means and SSA,” Scientific Reports, vol. 11,

3. A. Hamid, K. Gagliano, S. Rahman, N. Tulin, V. Tchiong, I. Obeid, and J. Picone, “The temple university artifact corpus: An annotated corpus of EEG artifacts,” in 2020 IEEE Signal Processing in Medicine and Biology Symposium (SPMB), pp. 1–4, 2020. ISSN: 2473-716X.