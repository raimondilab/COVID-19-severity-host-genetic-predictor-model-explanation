# COVID-19-severity-host-genetic-predictor-model-explanation
### Introduction

In this study, we identified 16 candidate genetic variants that are likely determinants of COVID-19 severity outcomes in patients from a prior study we carried out using 
2000 cohort genetic and clinical datasets (https://doi.org/10.21203/rs.3.rs-1062190/v1). We first used the saved ensemble host COVID-19 severity predictor model 
we trained with the 2000 cohort dataset and carried out external prediction to validate the model on a follow-up 3000 cohort dataset. 
Secondly, we further utilized the explainer dashboard python library to carry out an explanation of our model predictions particularly considering the new cohort 
dataset information. Lastly, we implored the Phenome-wide Association technique by leveraging the Bioinformatics tool of OpenTarget web tool (an open-source web
browser Bioinformatics tool) integrated in the model explanation part with explainerdashboard (SHAP sorted feature importance bar chart representation) the identified genetic variants with disease traits which could lead to plausible clinical trajectories of the COVID-19 disease in patients. 

In this repository, we built an interactive dashboard for predictions and explanations using a Whole Exome Sequencing (WES) and clinical covariates (age and gender) dataset of 3000 cohort follow-up European descent positive SARS-CoV-2 patients. We first validate the voting ensemble host genetic severity predictor (HGSP) model we developed from the 2000 cohort study and carried out further interpretations and post-hoc model explanations using the Explainerdasboard python library.  The Jupyter notebook (COVID-19 Host Genetic Predictor Model.ipynb) and this readme provide the codes and all the relevant information about the HGSP model and post-hoc model explanations. The HGSP model and Jupyter notebook script were further complied as a python script and used to develop the "COVID-19 Host Genetic Severity Predictor Model Web App". See the folder "COVID-19-severity-host-predictor-model-explanation-main" of this repository for more details on how to use the HGSP model web app. 
