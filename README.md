# GeoModelUncertainty - Assessing the uncertainty in CME geometric models due to solar wind structure.

## Introduction
This repository provides the models, analysis code, and data used to investigate the performance of CME geometric models. The HUXt solar wind model is used to simulate the evolution of 3 CME scenarios through 100 different background solar wind solutions. Synthetic Heliospheric Imager observations of the elongation of the CME flanks are used to estimate each CME's kinematics with a suite of geometric models, including ElEvoHI. We then analyse the errors in the geometrically modelled kinematics as a function of CME scenario and observer location, using the HUXt simulations as a ground truth. This study is under review for publication in AGU's Space Weather. 

## Installation
This project is written in Python 3.7.3, and the specific dependencies are listed in the ``requirements.txt`` and ``environment.yml`` files. 

After cloning or downloading ``GeoModelUncertainty``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where it is installed.

The simplest way to work with ``GeoModelUncertainty`` in ``conda`` is to create its own environment. With the anaconda prompt, in the root directory of ``GeoModelUncertainty``, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate geomodeluncertainty
``` 
Then the study can be reproduced by running the ``GeoModelUncertainty.py``.
```
>>python code/GeoModelUncertainty.py
```
The complete study should be expected to take up to 4 hours to run on a standard laptop. The production of the database of simulations and tracking the elongation of the CME flanks takes most of this time. After these files are produced, the remaining analysis is much quicker. 

## Contact
Please contact [Luke Barnard](https://github.com/lukebarnard). 

## Citation
Our article based on this analysis (Barnard et al. 2021) is under review for publication in AGU's Space Weather. A preprint of our article is available from ESSOAr at [https://doi.org/10.1002/essoar.10507552.1](https://doi.org/10.1002/essoar.10507552.1).

## License
This work is provided under the [MIT license](LICENSE.md).

