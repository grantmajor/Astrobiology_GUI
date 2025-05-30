## Load Necessary Libraries
# House Keeping Libraries
import streamlit as st
import numpy as np
from numpy import mean, std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning Libraries and Modules
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import svm, metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, \
    ConfusionMatrixDisplay, root_mean_squared_error, mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier

# Set Layout of the Application
st.set_page_config(layout="wide")

# Main Page Set Up
st.title("Teachable Artificial Intelligence for Astrobiology Investigations (AI$^{2}$)")
tab1, tab2, tab3, tab4 = st.tabs(["About",
                                  "Data and Preprocessing",
                                  "Unsupervised Learning",
                                  "Supervised Learning"]
                                 )

with tab1:
    st.markdown(
        "This application is designed to allow astrobiologists to experiment with machine learning approaches including unsupervised and "
        "supervised methods using their own data."
        " This application will be especially useful for those who may be interested in applying machine learning but either don't know "
        "where to start or do not have the time to learn programming languages.")

    st.markdown('Developed by Floyd Nichols and Grant Major')
    st.markdown('email: floydnichols@vt.edu\n\n'
                'email: grantmajor@vt.edu')

with tab2:
    st.subheader("Upload a Data File")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.error('Need to upload a file')
    else:
        def load_data(data_file, nrows):
            data = pd.read_csv(data_file, nrows=nrows)
            return data


        data_load_state = st.text('Loading data...')
        data = load_data(uploaded_file, 10000)
        data_load_state.text("Done!")

        if st.checkbox('Show Data'):
            st.subheader('Data')
            st.dataframe(data=data)

    st.divider()
    # Access and Download Example Data
    st.subheader(
        'If you do not have data of your own, use the following link to access available training sets from NASA AI Astrobiology')
    st.link_button('Go to NASA AI Astrobiology', 'https://ahed.nasa.gov/')

with tab3:
    col1, col2 = st.columns([1, 1])

    # Test code to make sure that a data file is uploaded before continuing
    try:
        data = data
    except:
        st.error("Please make sure that a data file is uploaded")
        st.stop()

    # Construct Dimensionality Reduction and Clustering
    with col1:
        st.subheader('**Here, the user can employ dimensionality reduction and clustering methods.**')
        st.divider()

        # Remove Columns that are Strings
        X = data.select_dtypes(include=['int64', 'float64'])

        st.subheader('Data and Hyperparameter Selection')

        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X = X.dropna()

        elements = st.multiselect("Select Explanatory Variables (default is all numerical columns):",
                                  X.columns,
                                  default=X.columns
                                  )

        y = data
        y = y.dropna()

        target = st.selectbox('Choose Target',
                              options=y.columns,
                              )

        options = st.selectbox(label='Select Dimensionality Reduction Method',
                               options=['Standard PCA',
                                        't-SNE'])

        # t-SNE Construction
        st.divider()
        # Set random state of the subsequent scripts
        np.random.seed(42)

        if options == 't-SNE':
            st.subheader('Define t-SNE Parameters')

            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )
            perplexity = st.number_input('Insert Perplexity',
                                         min_value=2
                                         )
            tsne = TSNE(n_components,
                        random_state=42,
                        perplexity=perplexity,
                        n_jobs=-1,
                        method='exact',
                        max_iter=5000
                        )
            tsne_result = tsne.fit_transform(X)
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0],
                                           'tsne_2': tsne_result[:, 1]}
                                          )

            # DBSCAN
            st.divider()
            clusters = st.selectbox(label='Select Cluster Method',
                                    options=['Kmeans',
                                             'DBSCAN',
                                             'Target'])

            if clusters == 'Kmeans':
                st.subheader('Define K-means Parameters')
                n_clusters = st.number_input('Enter Number of Clusters',
                                             min_value=2
                                             )

                X_Kmeans = KMeans(n_clusters=n_clusters).fit(tsne_result)
                labels = X_Kmeans.labels_

            elif clusters == 'DBSCAN':
                st.subheader('Define DBSCAN Parameters')
                eps = st.number_input('Enter Eps',
                                      min_value=0.5
                                      )
                min_samples = st.number_input('Enter Minimum Samples',
                                              min_value=1
                                              )

                X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_result)
                DBSCAN_labels = X_DBSCAN.labels_
                DBSCAN_labels = DBSCAN_labels.astype(str)
                labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

            else:
                labels = y

            with col2:
                # Plot t-SNE Results
                st.subheader('t-Distributed Stochastic Neighbor Embedding')
                fig, ax = plt.subplots()
                fig = px.scatter(tsne_result_df,
                                 x='tsne_1',
                                 y='tsne_2',
                                 color=labels,
                                 title=options
                                 )
                fig.update_traces(
                    marker=dict(size=8,
                                line=dict(width=2,
                                          color='Black')
                                )
                )
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_aspect('auto')
                ax.legend('Cluster',
                          bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.subheader('Define Standard PCA Parameters')
            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )

            pca = PCA(n_components=n_components)
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('pca', pca)])
            Xt = pipe.fit_transform(X)

            with col1:
                # DBSCAN
                st.divider()
                clusters = st.selectbox(label='Select Cluster Method',
                                        options=['Kmeans',
                                                 'DBSCAN',
                                                 'Target'])
                if clusters == 'Kmeans':
                    st.subheader('Define K-means Parameters')
                    n_clusters = st.number_input('Enter Number of Clusters',
                                                 min_value=2
                                                 )

                    X_Kmeans = KMeans(n_clusters=n_clusters).fit(Xt)
                    labels = X_Kmeans.labels_

                elif clusters == 'DBSCAN':
                    st.subheader('Define DBSCAN Parameters')
                    eps = st.number_input('Enter Eps',
                                          min_value=0.5
                                          )
                    min_samples = st.number_input('Enter Minimum Samples',
                                                  min_value=1
                                                  )

                    X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(Xt)
                    DBSCAN_labels = X_DBSCAN.labels_
                    DBSCAN_labels = DBSCAN_labels.astype(str)
                    labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

                else:
                    labels = y

            with col2:
                # Plot PCA Results
                st.subheader('Principal Component Analysis')
                fig, ax = plt.subplots()
                PCA_df = pd.DataFrame({'PCA_1': Xt[:, 0],
                                       'PCA_2': Xt[:, 1],
                                       'labels': labels},
                                      )
                fig = px.scatter(PCA_df,
                                 x='PCA_1',
                                 y='PCA_2',
                                 color=labels,
                                 title=options)
                fig.update_traces(
                    marker=dict(size=12,
                                line=dict(width=2,
                                          color='Black')
                                )
                )
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_aspect('auto')
                ax.legend(bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

                # Define and Plot Explained Variance Ratio
                fig, ax = plt.subplots()
                exp_var_pca = pca.explained_variance_ratio_
                fig = px.bar(exp_var_pca,
                             x=range(0, len(exp_var_pca)),
                             y=exp_var_pca,
                             title='PCA Explained Variance Ratio')

                fig.update_traces(
                    marker=dict(color='grey',
                                line=dict(width=3,
                                          color='Black')
                                )
                )

                fig.update_layout(
                    xaxis_title='Principal Component Index',
                    yaxis_title='Explained Variance Ratio'
                )
                ax.set_aspect('auto')
                ax.legend(bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('**Here, the user can employ regression and classification methods.**')
        st.divider()

        # Remove Columns that are Strings
        X_sup = data.select_dtypes(include=['int64', 'float64'])

        st.subheader('Data and Hyperparameter Selection')

        X_sup = X_sup.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X_sup = X.dropna()

        options_sup = st.selectbox(label='Select Prediction Type',
                                   options=['Classification',
                                            'Regression'])




        elements_sup = st.multiselect("Select Explanatory Variables (default is all numerical columns):",
                                      X_sup.columns,
                                      placeholder='Choose Option',
                                      default=X_sup.columns,
                                      )
        y_sup = data
        y_sup = y_sup.dropna()

        ohe_toggle = st.checkbox(label='Enable One-Hot-Encoding')

        if ohe_toggle:
            cat_features = data.select_dtypes(exlude=[np.number])
            ohe = OneHotEncoder(categories = 'auto',
                                sparse_output = False,
                                handle_unknown = 'error',

                                )

        #TODO: Make it so that the last column is the target variable automatically
        target_sup = st.selectbox('Choose Target',
                                  options=y_sup.columns,
                                  placeholder='Choose Option'
                                  )

        #setting variables to None to prevent errors from undeclared variables:
        class_algorithim = None
        X_predictions = None
        predicting_data = None

        X_sup = X_sup[elements_sup]
        y_sup = y_sup[target_sup]

        if target_sup in X_sup:
            st.warning("Overlapping target and explanatory variables detected.")

        #BEGIN TRAIN TEST SPLIT SECTION -------------------------------------------------------------------
        train_proportion = st.number_input('Enter the Proportion of Data to be Allocated to Training.',
                                           min_value=0.0,
                                           value=0.75,
                                           step=0.01,
                                           format="%.2f")
        X_train, X_test, y_train, y_test = train_test_split(X_sup, y_sup, train_size=train_proportion)

        st.divider()
        #END TRAIN TEST SPLIT SECTION ---------------------------------------------------------------------

        #BEGIN REGRESSION CODE: ------------------------------------------------------------------------------------------
        if options_sup == "Regression" and not X_sup.empty:
            #set to 0 to prevent errors when non quantile loss function is chosen
            quantile_value = 0
            reg_algorithim = st.selectbox(label='Chose Regression Algorithm',
                                          options=['Histogram-based Gradient Boosting Regressor Tree',
                                                   'Random Forest Regressor',
                                                   'Ridge Regressor'])

            #BEGIN HISTGRADBOOST CODE ------------------------------------------------------------------------------------
            if reg_algorithim == 'Histogram-based Gradient Boosting Regressor Tree':
                loss_function = st.selectbox(label='Choose Loss Function',
                                             options=['Squared Error',
                                                      'Absolute Error',
                                                      'Gamma',
                                                      'Poisson',
                                                      'Quantile'],
                                             index=0)

                if loss_function == 'Squared Error':
                    loss_function = 'squared_error'
                elif loss_function == 'Absolute Error':
                    loss_function = 'absolute_error'
                else:
                    loss_function = loss_function.lower()

                if loss_function == 'quantile':
                    quantile_value = st.number_input(label='Enter Quantile Value',
                                                     min_value=0.0,
                                                     max_value=1.0,
                                                     step=0.01,
                                                     format='%.2f')

                learn_rate = st.number_input(label='Enter Learning Rate',
                                             min_value=0.0,
                                             max_value=1.0,
                                             step=0.01,
                                             value=0.1,
                                             format=
                                             '%.2f')

                max_num_iter = st.number_input(label='Enter Maximum Number of Iterations',
                                               min_value=1,
                                               step=1,
                                               value=100)

                max_leaf = st.number_input(label='Enter Maximum Number of Leaves for Each Tree',
                                           min_value=2,
                                           value=31,
                                           step=1)

                selected_model = HistGradientBoostingRegressor(loss=loss_function,
                                                               quantile=quantile_value,
                                                               learning_rate=learn_rate,
                                                               max_iter=max_num_iter,
                                                               max_leaf_nodes=max_leaf
                                                               )

            #END HISTGRADBOOST CODE ----------------------------------------------------------------------------------

            #BEGIN RANDOM FOREST REGRESSOR CODE ----------------------------------------------------------------------
            elif reg_algorithim == 'Random Forest Regressor':
                num_estimators = st.number_input(label='Enter the number of estimators.',
                                                 min_value=1,
                                                 step=1,
                                                 value=100)

                selected_criterion = st.selectbox(label='Select a criterion',
                                                  options=['Squared Error', 'Absolute Error', 'Friedman MSE',
                                                           'Poisson'])

                #casts the selected criterion into the correct format for scikit
                if selected_criterion == 'Squared Error':
                    selected_criterion = 'squared_error'
                elif selected_criterion == 'Absolute Error':
                    selected_criterion = 'absolute_error'
                elif selected_criterion == 'Friedman MSE':
                    selected_criterion = 'friedman_mse'
                else:
                    selected_criterion = selected_criterion.lower()

                num_min_samples_split = st.number_input(
                    "Enter the minimum number of samples required to split an internal node",
                    min_value=2,
                    step=1,
                    value=2)

                enable_tree_depth = st.checkbox('Enable tree depth parameter',
                                                value=False)

                if enable_tree_depth:
                    tree_depth = st.number_input('Enter the maximum depth of each tree.',
                                                 min_value=1,
                                                 step=1)
                    selected_model = RandomForestRegressor(n_estimators=num_estimators,
                                                           criterion=selected_criterion,
                                                           max_depth=tree_depth,
                                                           min_samples_split=num_min_samples_split
                                                            )
                else:
                    selected_model = RandomForestRegressor(n_estimators=num_estimators,
                                                           criterion=selected_criterion,
                                                           min_samples_split=num_min_samples_split
                                                            )

            #END RANDOM FOREST REGRESSOR CODE -------------------------------------------------------------------------

            #BEGIN RIDGE REGRESSOR CODE -------------------------------------------------------------------------------
            elif reg_algorithim == 'Ridge Regressor':
                alpha_value = st.number_input(label='Input Alpha Value',
                                              min_value=0.0,
                                              value=1.0,
                                              step=0.01,
                                              format='%.2f')

                selected_model = Ridge(alpha=alpha_value)

            #END RIDGE REGRESSOR CODE ---------------------------------------------------------------------------------

        #END REGRESSION CODE ---------------------------------------------------------------------------------------

        #BEGIN CLASSIFICATION CODE ------------------------------------------------------------------------------------
        elif options_sup == 'Classification' and not X_sup.empty:
            class_algorithim = st.selectbox(label='Choose Classification Algorithm',
                                            options=['Support Vector Machine (SVM)',
                                                     'k-Nearest Neighbors (k-NN)',
                                                     'Random Forest Classifier']
                                            )

            #BEGIN SUPPORT VECTOR MACHINE CODE -----------------------------------------------------------------------
            if class_algorithim == 'Support Vector Machine (SVM)':
                #Sets degree to a default value in case kernel_type isn't polynomial and thus degree isn't declared
                degree = 3

                #Selecting C value hyperparameter
                c_value = st.number_input('Input C Value',
                                          min_value=0.0,
                                          value=1.0,
                                          step=0.01,
                                          format="%.2f")

                #Selecting Kernel hyperparameter
                kernel_type = st.selectbox('Select a Kernel',
                                           ('Linear', 'Polynomial', 'Radial Basis Function'),
                                           index=0)

                #Selecting Degree hyperparameter
                if kernel_type == 'Polynomial':
                    degree = st.number_input('Enter a degree', min_value=0)

                #changes the kernel_type var to a valid value for the svm function
                if kernel_type == 'Linear':
                    kernel_type = 'linear'
                elif kernel_type == 'Polynomial':
                    kernel_type = 'poly'
                else:
                    kernel_type = 'rbf'

                # creates svm model using inputted values
                selected_model = svm.SVC(C=c_value,
                                         kernel=kernel_type,
                                         degree=degree)
            #END SUPPORT VECTOR MACHINE CODE ----------------------------------------------------------------------

            #BEGIN K-NEAREST NEIGHBORS CODE -----------------------------------------------------------------------
            elif class_algorithim == 'k-Nearest Neighbors (k-NN)':

                k_value = st.number_input('Input K Value.',
                                          min_value=1,
                                          value=1)

                selected_model = KNeighborsClassifier(n_neighbors=k_value)

            #END K-NEAREST NEIGHBORS CODE -------------------------------------------------------------------------

            #BEGIN RANDOM FOREST CLASSIFIER CODE -------------------------------------------------------------------
            elif class_algorithim == 'Random Forest Classifier':

                num_estimators = st.number_input('Enter the Number of Estimators.',
                                                 min_value=1,
                                                 step=1,
                                                 value=100)

                selected_criterion = st.selectbox('Select a Criterion',
                                                  ('Gini', 'Entropy', ' Log Loss'))
                if selected_criterion == "Log Loss":
                    selected_criterion = "log_loss"
                else:
                    selected_criterion = selected_criterion.lower()

                num_min_samples_split = st.number_input(
                    "Enter the Minimum Number of Samples Required to Split an Internal Node",
                    min_value=2,
                    step=1,
                    value=2)

                enable_tree_depth = st.checkbox('Enable Tree Depth',
                                                value=False)

                if enable_tree_depth:
                    tree_depth = st.number_input('Enter the Maximum Depth of Each Tree.',
                                                 min_value=1,
                                                 step=1)
                    selected_model = RandomForestClassifier(n_estimators=num_estimators,
                                                            criterion=selected_criterion,
                                                            max_depth=tree_depth,
                                                            min_samples_split=num_min_samples_split
                                                            )
                else:
                    selected_model = RandomForestClassifier(n_estimators=num_estimators,
                                                            criterion=selected_criterion,
                                                            min_samples_split=num_min_samples_split
                                                            )

            #END RANDOM FOREST CLASSIFIER CODE ------------------------------------------------------------------

        #END CLASSIFICATION CODE ------------------------------------------------------------------------------------
    if not X_sup.empty and not y_sup.empty:
        selected_model.fit(X_train, y_train)

    #BEGIN MODEL METRICS CODE -------------------------------------------------------------------------------
    with col2:
        show_metrics_enabled = False

        # check box for showing model metrics
        if not X_sup.empty:
            y_pred = selected_model.predict(X_test)
            show_metrics_enabled = st.checkbox("Show Model Metrics")
        else:
            st.warning("Select explanatory variables to continue.")

        if show_metrics_enabled and options_sup == 'Classification':
            st.header("Model Performance Metrics")

            #BEGIN CLASSIFICATION REPORT CODE --------------------------------------------------------
            class_report = classification_report(y_test, y_pred, output_dict = True)
            st.subheader("Classification Report")
            class_report = pd.DataFrame(class_report).transpose()
            st.table(class_report)

            #END CLASSIFICATION REPORT CODE -----------------------------------------------------------

            #BEGIN CONFUSION MATRIX CODE --------------------------------------------------------------
            #Creates confusion matrix
            # Compute matrix

            st.subheader('Confusion Matrix')
            conf_mat = confusion_matrix(y_test, y_pred)

            # Define custom labels
            conf_mat_labels = np.array([
                f'True Negative\n{conf_mat[0, 0]}',
                f'False Positive\n{conf_mat[0, 1]}',
                f'False Negative\n{conf_mat[1, 0]}',
                f'True Positive\n{conf_mat[1, 1]}'
            ]).reshape(2, 2)

            # Reset any existing figures
            plt.close('all')

            # Create themed figure
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')


            sns.heatmap(conf_mat,
                        annot=conf_mat_labels,
                        fmt='',
                        cmap='Blues',
                        cbar=True,
                        linewidths=0.5,
                        linecolor='#0e1117',
                        ax=ax)

            # Style text and axes
            ax.set_title('Confusion Matrix', color='white')
            ax.set_xlabel('Predicted', color='white')
            ax.set_ylabel('Actual', color='white')
            ax.tick_params(colors='white', labelsize=10)

            # Style colorbar ticks
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

            fig.tight_layout()
            st.pyplot(fig)

            st.divider()
        #END CONFUSION MATRIX CODE -----------------------------------------------------------------


        elif show_metrics_enabled and options_sup == 'Regression':
            try:
                check_is_fitted(selected_model)
            except NotFittedError as e:
                st.error('Model has not been fitted. Model metrics cannot be calculated.')

            # BEGIN REGRESSION LOSS FUNCTION CODE --------------------------------------------------------

            y_pred = selected_model.predict(X_test)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = root_mean_squared_error(y_test, y_pred)
            MAE = mean_absolute_error(y_test, y_pred)
            MAPE = mean_absolute_percentage_error(y_test, y_pred)

            reg_metrics = pd.DataFrame([{'Mean Squared Error': MSE,
                                              'Root Mean Squared Error': RMSE,
                                              'Mean Absolute Error': MAE,
                                              'Mean Absolute Percentage Error': MAPE}])

            st.dataframe(reg_metrics, hide_index=True)

            #END REGRESSION LOSS FUNCTION CODE ------------------------------------------------------------


            #BEGIN ACTUAL v PREDICTED GRAPH CODE --------------------------------------------------------------
            fig, ax = plt.subplots()
            fig.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            # Scatter and reference line
            ax.scatter(y_test, y_pred, color='deepskyblue', edgecolor='white', s=60, alpha=0.8)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', lw=2, label='Perfect Prediction')

            # Add legend and set text color manually
            legend = ax.legend(facecolor='#0e1117', frameon=False)
            for text in legend.get_texts():
                text.set_color('white')

            # Styling
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='white', labelsize=10)
            ax.set_title('Actual vs. Predicted', color='white')
            ax.set_xlabel('Actual', color='white')
            ax.set_ylabel('Predicted', color='white')

            st.pyplot(fig)

            #END ACTUAL V PREDICTED GRAPH CODE ------------------------------------------------------------------


        #BEGIN PREDICTION UPLOAD CODE --------------------------------------------------------------
        #Reads in data file that user wants predictions on
        if not X_sup.empty:
            predicting_data_file = st.file_uploader('Upload a file with values to be predicted.')

            if predicting_data_file is not None:
                data_load_state2 = st.text('Loading data....')
                predicting_data = load_data(predicting_data_file, 10000)
                data_load_state2.text('Done!')

        #END PREDICTION UPLOAD CODE ---------------------------------------------------------------
