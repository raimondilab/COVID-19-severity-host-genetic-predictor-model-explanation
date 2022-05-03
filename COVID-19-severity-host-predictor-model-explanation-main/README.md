# COVID-19-severity-host-predictor-model-explanation
COVID-19 Host Genetic Predictor Model and Explanation web app will be deploy to Heroku shortly

This web app predicts patients' COVID-19 severity using 16 candidate genetic variants (Fully supported variants we identified from a prior study of 2000 cohort patients' Whole Exome Sequencing) and clinical covariates (age and gender) datasets. 
The 18 features (16 variants and 2 covariates) were used as input features (age, gender, ZBED3(rs531117283), PLEC(rs140300753), TRIM72(), HDGFL2(rs146793578), SECISBP2L(rs75595801),
CEP131(rs2659015), GOLGA6L3(rs367838829), PCSK5(rs72745135), GFM1(rs370496368), ZBTB3(rs544641), BMS1P1;FRMPD2B(), SPATA6(rs77303590), CNTFR(), MIR933(rs79402775), 
ZRANB3(rs1465146591), LOC100996720()). The output (target) variable is COVID-19 severity (a binarized outcome variable with severe patients coded as 1 and asymptomatic patients coded as 0).

The COVID-19 Host Genetic Predictor Model and Explanation web app was built in python using the following libraries:
- streamlit
- pandas 
- numpy
- scikit-learn
- joblib
- explainerdashboard
- matplotlib
- seaborn
