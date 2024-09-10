'''
Author       : wyx-hhhh
Date         : 2024-09-05
LastEditTime : 2024-09-09
Description  : Streamlit app for visualizing model results with multi-selection capability.
'''

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import pandas as pd
import streamlit as st
from utils.vis_utils import ResultsManager


def intro():
    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown("""
        123
    """)


def results_visualization():
    st.write("# Results Visualization")

    # Select multiple models on the sidebar
    st.sidebar.write("## Select Models")
    results_manager = ResultsManager()
    selected_models = st.sidebar.multiselect("Models", results_manager.model_list)

    if selected_models:
        st.write("## Model Metrics")
        combined_metrics = pd.DataFrame()

        for model_name in selected_models:
            results_manager.select_model(model_name)
            results_manager.extract_metrics()
            df = results_manager.to_dataframe()

            if df.empty:
                st.write(f"No metrics found for model: {model_name}")
            else:
                st.write(f"### Metrics for Model: {model_name}")
                st.dataframe(df)
                combined_metrics = pd.concat([combined_metrics, df], ignore_index=True)

        # Display combined metrics for selected models
        if not combined_metrics.empty:
            st.write("## Combined Metrics for Selected Models")
            st.dataframe(combined_metrics)

        # View files associated with selected models
        st.write("## View Files")
        run_info = st.sidebar.selectbox("Run Info", results_manager.runinfo_file_name)
        st.write(results_manager.get_run_info_data(run_info))

        # Cleanup unreferenced files
        st.write("## Delete Unreferenced Files")
        if len(selected_models) > 1:
            st.write("Please select only one model to delete unreferenced files.")
        else:
            unreferenced_files = results_manager.clean_up()
            if unreferenced_files:
                st.write(unreferenced_files)
                if st.button("Delete Unreferenced Files"):
                    results_manager.delete_unreferenced_files()
                    unreferenced_files = []
                    st.success("Unreferenced files deleted.")
    else:
        st.write("Please select at least one model.")


page_names_to_funcs = {"—": intro, "Results": results_visualization}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
