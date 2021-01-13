
This project aims to investigate the effectiveness of Signature Methods to extract predictive and meaningful time series features from Electronic Healthcare Records (EHR) for heart failure (HF) disease early diagnosis. The project will investigate the task using data from the Clinical Practice Research Datalink (CPRD) linked to Biobank data using event based EHR data from routinely collected primary and secondary care providers. 

This is the initial release will be followed with additional makefiles to demonstrate greater reproducibility.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── experiments        <- python experiment files for running on cluser
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. For some report figures.
    │                         
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

