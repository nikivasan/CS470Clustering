# CS470FinalProject
**By Niki Vasan, David Gaviria, Gianluis Hernandez**

**Collaboration Statement**: This project was conducted by the three co-authors of this paper, with consultation by our TA Hope Mumme. No other sources or peers were consulted outside of relevant documentation.

### Introduction
This repository contains the code and files from our CS470 Data Mining course final project. The goal of the project was to work collaboratively in teams to apply data mining techniques to real-world tasks. Our group's goal was to cluster Spotify song data based on quantitative attributes and then compare the genre profiles of each cluster. We had three main deliverables:
1. Code Repository: Train, and evaluate model, push code + relevant files to github repository 
2. Final Presentation: 5 minute presentation explaining model design, evaluation and results
3. Report: report detailing problem statement, data description, methods, results and key findings of our project

### Repo Structure
* **data_preprocessing**: contains original and preprocessed datasets as well as the notebooks used for pre-processing
* **model**: contains code used to build and run model
* **assessment**: plots and code used to evaluate model 
* **reports**: contains our written report (in pdf form with Latex source code) and slide deck

### Relevant Links and Info
To run our code, navigate to the [model](https://github.com/nikivasan/CS470FinalProject/tree/main/model) directory, and run the `main.py` file by typing `python main.py genre_processed_v2.csv.` This will run both K-Means and DB-Scan with pre-selected optimal parameters. 

Click [here](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify?resource=download) to download the dataset we used, or download the data directly from our [repo](https://github.com/nikivasan/CS470FinalProject/tree/main/data_preprocessing/original_datasets). 

To watch our presentation, click [here](https://emory.zoom.us/rec/play/X6X2qz4OzMnwQbZL_MWDNOKUOrBO0IzN_tW9-GTsgvjo3SVCOp7fP5927CDbhnsJJFqCtGrToz6UpbIM.q5vRSOYoU5iFUV9a?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Femory.zoom.us%2Frec%2Fshare%2FamvwoM64jhVxoEVkQUsiIBsgqV0QMwFJwcnGPdOVyevKfQX3oUb2dmO8s7M_g9Mg.kNvJRGd1MZjulBZB).
