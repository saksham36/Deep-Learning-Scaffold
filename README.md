# Deep Learning Training Scaffold

This repository contains a starter app which can be used to speed-up your next machine learning/deep learning project.

## Requirements

Feel free to update your `environment.yml` file. I've provided one example file. To create the environment run `conda env create -f environment.yml`

### File Structure
```
- utils/
  - __init__.py         : Needed for python packaging for `utils`
  - dataset.py          : Script to create dataset
  - filtering.py        : Script for filtering. Needed for raw data processing
  - collate_data.py     : Script for collating raw data into structured data
  - general_tools.py    : General tools for arguments, and backup while training
  - specific_tools.py   : Specific tools for given application
  - plotting.py         : Some plotting scripts to plot results and data
  - train_tools.py      : Tools for training
- data/                 : Place to store data
- logs/                 : Place where logs will be saved
- model.py              : GRU Model
- main.py               : Main script
- trainer.py            : Script for trainer
- __init__.py           : Needed for python packaging
- environment.yml       : Dependencies for environment

- LICENSE
- README.md
```
## To Run

`python main.py <args>`
