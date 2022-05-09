# COVID-19-severity-host-predictor-model-explanation
Here is a short FAQ about how to use the COVID-19 Host Genetic Severity Predictor Model Web Application

### Introduction 

In this repository, we built an interactive dashboard for a voting ensemble host genetic severity predictor (HGSP) model we developed with 16 fully supported genetic variants together with clinical covariates (age and gender) identified from a prior study of the Whole Exome Sequencing dataset from 2000 cohort European descent SARS-CoV-2 positive patients. We further utilized the ExplainerDashboard library in Python for the post-hoc model interpretations and explanations for the 3000 cohort follow-up dataset. The Jupyter notebook (.ipynb) and this readme provide the codes and all the relevant information about this web App.

1) What is the COVID-19 Host Genetic Severity Predictor Model Web App about?

1a) This is a python web application predictor model developed based on a study we carried out using Whole Exome Sequencing (WES), clinical covariate (age and gender), and clinical COVID-19 severity outcome dataset information of 2000 cohort European descent patients. 
This web app predicts patients' COVID-19 severity using 16 candidate genetic variants (Fully supported variants we identified from a prior study of 2000 cohort patients' Whole Exome Sequencing) and clinical covariates (age and gender) datasets. The 18 features (16 variants and 2 covariates) were used as input features (age, gender, ZBED3(rs531117283), PLEC(rs140300753), TRIM72(), HDGFL2(rs146793578), SECISBP2L(rs75595801), CEP131(rs2659015), GOLGA6L3(rs367838829), PCSK5(rs72745135), GFM1(rs370496368), ZBTB3(rs544641), BMS1P1;FRMPD2B(), SPATA6(rs77303590), CNTFR(), MIR933(rs79402775), ZRANB3(rs1465146591), LOC100996720()). 
The output (target) variable is COVID-19 severity (a binarized outcome variable with severe patients coded as 1 and asymptomatic patients coded as 0).
The ensemble model proposed by our study was gotten from the combinations of decision tree models (Random Forest and XGBoost classifier) trained on a 5-fold CV splitting strategy of 16 fully supported variants (i.e., 16 non-zero weighted variants consistent across the 5-fold CV pool of variants we retrained with decision tree models from the 2000 cohort prior study) and covariates (age and gender). We carry out external validation of the saved ensemble model from the 2000 cohort studies, we used a follow-up 3000 cohort WES dataset information. We further explore various model interpretability and explanations for us to further shed light on the complex genetic interactions that might interplay with the severity outcome of the COVID-19 disease in patients. The Python library “Explainerdashboard” was further used for the model explanations with the incorporation of disease-traits associations for potential identifications of the genetic determinants of COVID-19 clinical trajectories.  

1b) How can I use the COVID-19 Host Genetic Severity Predictor (HGSP) Model? 

First, the user(s) required a WES and clinical covariate (clinical severity outcome grading, age, and gender) dataset. The user(s) will need to curate the 16 variants from the WES dataset, binarize the grading outcome variable (group 1 (severe) = grading 5+4+3 versus group 0 = grading 0), and together with the covariates (age and gender) will be used to develop the feature matrix. There were 18 independent input features and one output variable (patients’ grouping). Upon access to the WebApp, the user(s) are guided with an example dataset in CSV format and a brief video link to the 2000 cohort study. The user(s) drags and drops the CSV file and clicks the sidebar button to select “model prediction” or “explanation”. We provided these options should in case a user may not be interested in the model prediction and will rather prefer to see the model explanation or vice versa. If the model prediction option is selected for example the user will further select the different performance evaluation metrics (Confusion matrix plot, precision-recall curve, ROC-AUC curve). The performance metrics (accuracy, precision, and recall scores) will pop out alongside the plots. If the user seeks further explanations, can go ahead to click the “explanation” button to see the explanation of the model via the explainer dashboard approach. However, the results for the explanation will take some time if there are many samples (rows) due to the explainer dashboard using the SHAP permutation explanation approaches to calculate the feature importance, hence, perturbing individual’s explanations, and other performance metrics. The explainer dashboard URL is built into the WebApp to display the rightful explanations for the users without having to run it locally on their computer. Users can save the visualization metrics of interest in PDF, Html, JPEG, or PNG file formats.

2) What is ExplainerDashboard?
ExplainerDashboard is a library for quickly building interactive dashboards for analyzing and explaining the predictions and workings of (scikit-learn compatible) machine learning models, including random forest, xgboost, catboost, and lightgbm. This makes your model transparent and explainable with just two lines of code. It is a python package that generates interactive dashboards which allow users to understand as well as explain how the model works and how it is deciding the outcome. Without such a tool, a machine learning model is a “Black Box model”. Hence, it becomes difficult to explain the reason behind the decision made by the model and which factors are impacting its decision-making.

3) Which libraries have been used for the creation of the dashboard?
In developing the host genetic severity predictor web application, we used the following libraries:
-	streamlit==0.71.0
-	pandas==1.1.3
-	numpy==1.19.2
-	pillow==8.0.1
-	plotly==4.14.1
-	scikit-learn==0.23.2
-	seaborn==0.11.1
-	matplotlib==3.3.3
-	xgboost==1.1.1
-	explainerdashboard==0.3.8.2
-	joblib==1.2.0
 Using these packages in Python, we deployed the HGSP model and the post-hoc model explanation using the explainer dashboard as a web app in Heroku, Jupyter, or Colab notebooks.
 
4) Which dataset has been used?
We are using the WES dataset feature_count_16_full_supported_variants_latest_3000_cohort_arranged_updated.csv' to train the HGSP model and build the interactive post-hoc model explanation dashboard. Details about the HGSP model and the study design and model setup are discussed are found at https://clinicaltrials.gov/ct2/show/NCT04549831, https://doi.org/10.21203/rs.3.rs-1062190/v1.

#### Key                                                                     Meaning

COVID_19_host_genetic_predictor_Xplainer_webapp.py = a python script of the HGSP model and post-hoc explanation using the explainer dashboard 

new_data_small_sample.csv = a small size (24 sample points) sample dataframe of 16 fully supported variants and covariates (age and gender) 

feature_count_16_full_supported_variants_latest_3000_cohort_arranged_updated = a large size (618 sample points) sample dataframe of 16 fully supported variants and covariates (age and gender)

ensemble_model_2000_cohort_16_full_support_with_covariates_1 = HGSP saved joblib model 

requirements.txt = requirements containing lists of all the python libraries used to develop the web app.

runtime.txt = a file added to the web app's root directory that declares the exact version number to use.

