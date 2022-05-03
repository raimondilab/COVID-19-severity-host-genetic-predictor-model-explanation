import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import streamlit.components.v1 as components
from sklearn.ensemble import VotingClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    #st.title("About")
    st.markdown('**About**')
    st.write('''We employed a multifaceted computational strategy to identify
    the genetic factors contributing to increased risk of severe COVID-19 infection
    from a Whole Exome Sequencing (WES) dataset of a cohort of 2000 Italian patients.
    We coupled a stratified k-fold screening, to rank variants more associated with severity,
    with training of multiple supervised classifiers, to predict severity on the basis of screened features.
    Feature importance analysis from decision-tree models allowed to identify a handful of 16 variants
    with highest support which, together with age and gender covariates, were found to be most predictive
    of COVID-19 severity. More details were shared@ https://doi.org/10.21203/rs.3.rs-1062190/v1.
    Watch the summary video below of our findings to know more about this WebApp'''
    )
    # Display a brief summary video of our web app.
    st.video("https://youtu.be/hdWwe95FUXw")
    # Displays the dataset
    st.subheader('Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to see example of WES CSV Dataset we used'):
            # Diabetes dataset
            #diabetes = load_diabetes()
            #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            #Y = pd.Series(diabetes.target, name='response')
            #df = pd.concat( [X,Y], axis=1 )

            #st.markdown('The Diabetes dataset is used as the example.')
            #st.write(df.head(5))

            # covid-91 fully supported dataset
            covid_WES_data = pd.read_csv("new_data_small_sample.csv", delimiter=',', quotechar='"', index_col='sample_ID')

            df = covid_WES_data
            #X = pd.DataFrame(df, columns=df.columns)
            #Y = pd.Series(boston.target, name='response')
            X = df.iloc[:,1:-1] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            Y = df.iloc[:,-1] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The fully supported 16 variants dataset is used as the example.')
            st.write(df.head(5))





    st.markdown("Will you like to make a prediction and also an explanation of genetic variants linked to Covid-19 severity? ")
    st.markdown("Click 'explainer' to view model explanation")

    st.sidebar.title("Covid-19 severity predictions")
    st.sidebar.markdown("Do you have a Whole Exome Sequencing (WES) genetic Dataset of Covid-19 patients? ")

@st.cache(persist = True)
def uploaded_file(df):
    df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    df.set_index('sample_ID', inplace=True)
def build_model(df):
    x = df.iloc[:,1:-1] # Using all column except for the last column as X
    y = df.iloc[:,-1] # Selecting the last column as Y
    class_name = ['Asymptomatic', 'Severe']
    st.markdown('**Validation set Dataset dimension**')
    st.write('x')
    st.info(x.shape)
    st.write('y')
    st.info(y.shape)
        #return x_train, x_test, y_train, y_test
    st.markdown('**Variable details**:')
    st.write('Predictors (All 16 full supported variants are shown)')
    st.info(list(x.columns[:20]))
    st.write('Target (outcome)')
    st.info(y.name)
    #st.markdown('**Model Prediction/explanations**')
    scale = StandardScaler()
    X_tr = pd.DataFrame(scale.fit_transform(x))
    X_tr.columns = x.columns.values
    X_tr.index = x.index.values
    x = X_tr

    y = label_binarize(y, classes=[0, 1])

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(ensemble_ML, x, y, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(ensemble_ML, x, y)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(ensemble_ML, x, y)
            st.pyplot()

    #df = load_data()
    #x_train, x_test, y_train, y_test = split(df)
    class_names = ['Non-severe', 'Severe']
    st.sidebar.subheader("Choose Prediction model")
    model = st.sidebar.selectbox("Select Model or Explanation", ("ensemble_model_2000_cohort_16_full_support_with_covariates_1", "explainer"))

    if model == "ensemble_model_2000_cohort_16_full_support_with_covariates_1":
        with open('ensemble_model_2000_cohort_16_full_support_with_covariates_1', 'rb') as f1:
            ensemble_ML = joblib.load(f1)
            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Predict", key = 'predict'):
                st.subheader("Ensemble Model prediction Results")
            #model.fit(x_train, y_train)
                accuracy = ensemble_ML.score(x, y)
                y_pred = ensemble_ML.predict(x)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y, y_pred, labels = class_names).round(2))
                st.write("Recall: ", recall_score(y, y_pred, labels = class_names).round(2))
                plot_metrics(metrics)

    if model == "explainer":
        if st.sidebar.button("Explanation", key = 'explain'):
            feature_descriptions = {"age" : "Age (Years)", "gender" : "Gender",
                        "ZBED3" : "ZBED3(rs531117283)[Hint: Genetic disease associated - BMI-adjusted waist-hip ratio, type 2 diabetes mellitus]",
                        "PLEC" : "PLEC(rs140300753) [Hint: Genetic disease associated - Abnormalities of breathing, Epidermolysis bullosa simplex with muscular dystrophy, Epidermolysis bullosa simplex with pyloric atresia]",
                        "TRIM72" : "TRIM72() [Hint: Genetic disease associated - breast adenocarcinoma, diabetes mellitus, coronary artery disease]",
                        "HDGFL2" : "HDGFL2(rs146793578) [Hint: Genetic disease associated - mean corpuscular hemoglobin concentration, BMI-adjusted waist-hip ratio, smoking status measurement]",
                        "SECISBP2L" : "SECISBP2L(rs75595801) [Hint: Genetic disease associated - calcium measurement, diastolic blood pressure, lung adenocarcinoma, alcohol consumption measurement]",
                        "CEP131" : "CEP131(rs2659015)[Hint: Genetic disease associated - serum IgG glycosylation measurement, brain glioblastoma, IgG fucosylation measurement, vitamin D measurement]",
                        "GOLGA6L3" : "GOLGA6L3(rs367838829) [Hint:Genetic disease - BMI-adjusted waist and hip circumference]",
                        "PCSK5" : "PCSK5(rs72745135) [Hint: Genetic disease associated - Abnormalities of breathing, Currarino Syndrome and Avian Influenza.]",
                        "GFM1" : "GFM1(rs370496368) [Hint:Genetic disease associated - Hepatoencephalopathy due to combined oxidative phosphorylation deficiency type 1, combined oxidative phosphorylation deficiency, systolic blood pressure]",
                        "ZBTB3" : "ZBTB3(rs544641)[Hint: Genetic disease associated - alcohol consumption measurement, wellbeing measurement, chronotype measurement]",
                        "BMS1P1;FRMPD2B" : "BMS1P1;FRMPD2B()", "SPATA6" : "SPATA6(rs77303590) [Hint: Genetic disease associated - urinary metabolite measurement, systolic blood pressure]",
                        "CNTFR" : "CNTFR() [Hint: Genetic disease associated - blood protein measurement, heel bone mineral density, adolescent idiopathic scoliosis]", "MIR933" : "MIR933(rs79402775)",
                        "ZRANB3" : "ZRANB3(rs1465146591)[Hint: Genetic disease associated - body height, gut microbiome measurement, body mass index]",
                        "LOC100996720" : "LOC100996720()"
                        }
            with open('ensemble_model_2000_cohort_16_full_support_with_covariates_1', 'rb') as f1:
                ensemble_ML = joblib.load(f1)
                explainer = ClassifierExplainer(ensemble_ML, x, y.ravel(),
                                descriptions=feature_descriptions, # adds a table and hover labels to dashboard
                                labels=['Asymptomatic', 'Severe'], # defaults to ['0', '1', etc]
                                idxs = df.index, # defaults to X.index
                                index_name = "sample_ID", # defaults to X.index.name
                                target = "grouping", # defaults to y.name
                                )
                db = ExplainerDashboard(explainer, title="An explainable model of host genetic interactions linked to COVID-19 severity", # defaults to "Model Explainer"
                                whatif=False, )
                #explanation = st.sidebar.multiselect("What Model explanation will you like to know?", ('metrics_list', 'dashboardurl'))
                #if st.sidebar.button("Model explanation", key = 'explain'):
                st.subheader("Individualistic Explanation")
                    # Display explainer HTML object
                #dashboardurl = 'http://192.168.43.195:8050'
                ExplainerDashboard(db).run()
                #st.components.v1.iframe(dashboardurl, width=None, height=900, scrolling=True)

def app():
    """ Set appearance to wide mode."""
    st.title("An explainable model of host genetic interactions linked to COVID-19 severity")

    #dashboardurl = 'http://192.168.43.195:8050'
    #dashboardurl = db.run(port=8501)
    #explainer.dump('explainer.joblib')
    #ClassifierExplainer.from_file('explainer.joblib')
    #st.components.v1.iframe(dashboardurl, width=None, height=900, scrolling=True)

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# An Explainable Model of Host Genetic interactions linked to COVID-19 severity WebApp

In this WebApp, an **ensemble model** we developed by training decision tree like models
(Random Forest and XGBoost classifiers from a 5-FoldCV splitting strategy)  and
**explainerdashboard** python library were used for predictions and explanation
of model's performance at individualistic level.

Developed by: [Anthony Onoja](https://github.com/Donmaston09), [Francesco Raimondi](https://github.com/raimondilab), [Mirco Nani](https://github.com/mirconanni)
""")

#---------------------------------#
# Sidebar - Collects featurecount input variant features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/Donmaston09/)
""")

# Sidebar - Specify parameter settings
#with st.sidebar.header('2. Set Parameters'):
   # split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
   # seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#
# Main panel

        #build_model(df)

if __name__ == '__main__':
    main()
