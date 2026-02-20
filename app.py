import json
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# --- Configuration & i18n ---
st.set_page_config(page_title="BioGenius: RNA-Seq", page_icon="🧬", layout="wide")


@st.cache_data
def load_translations(language_code: str) -> Dict[str, str]:
    """Loads and caches the localization JSON file based on selected language."""
    with open(f"locales/{language_code.lower()}.json", "r", encoding="utf-8") as f:
        return json.load(f)


lang = st.sidebar.radio("🌐 Language / Dil", ["EN", "TR"], horizontal=True)
t = load_translations(lang)

st.title(t["title"])
st.markdown(t["subtitle"])
st.markdown("---")


# --- Core Engine ---
@st.cache_data(show_spinner=False)
def run_deseq2(
        counts_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        design_factor: str,
        contrast_group: str,
        base_group: str
) -> pd.DataFrame:
    """
    Executes the PyDESeq2 pipeline for differential expression analysis.
    Returns a processed dataframe with log2FoldChange and adjusted p-values.
    """
    counts_df = counts_df.T

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata_df,
        design_factors=design_factor,
        n_cpus=4
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=[design_factor, contrast_group, base_group])
    stat_res.summary()

    res_df = stat_res.results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
    res_df = res_df.reset_index().rename(columns={'index': 'Gene'})

    # Prevent math domain errors during -log10 transformation
    res_df['padj'] = res_df['padj'].replace(0, 1e-300)
    res_df['-log10(padj)'] = -np.log10(res_df['padj'])

    return res_df


# --- UI Sidebar ---
with st.sidebar:
    st.header(t["sidebar_data"])
    counts_file = st.file_uploader(t["counts_file"], type=["csv"])
    meta_file = st.file_uploader(t["meta_file"], type=["csv"])

    st.markdown("---")
    st.markdown(t["no_data_msg"])
    if st.button(t["load_sample"]):
        try:
            st.session_state['counts_df'] = pd.read_csv("data/sample_counts.csv", index_col=0)
            st.session_state['meta_df'] = pd.read_csv("data/sample_meta.csv", index_col=0)
            st.success(t["sample_loaded"])
        except FileNotFoundError:
            st.error(f"{t['error']} 'data/' directory or sample files are missing.")

    st.markdown("---")
    st.header(t["sidebar_params"])
    lfc_threshold = st.slider(t["lfc_help"], 0.0, 3.0, 1.0, 0.1)
    padj_threshold = st.number_input(t["padj_help"], 0.001, 1.0, 0.05, 0.01)

# --- Data Resolution ---
counts_df, meta_df = None, None

if counts_file and meta_file:
    counts_df = pd.read_csv(counts_file, index_col=0)
    meta_df = pd.read_csv(meta_file, index_col=0)
elif 'counts_df' in st.session_state and 'meta_df' in st.session_state:
    counts_df = st.session_state['counts_df']
    meta_df = st.session_state['meta_df']

# --- Main Execution Flow ---
if counts_df is not None and meta_df is not None:
    try:
        st.subheader(t["config_header"])
        col1, col2, col3 = st.columns(3)

        design_col = col1.selectbox(t["design_col"], meta_df.columns)
        unique_conditions = meta_df[design_col].unique()

        contrast_group = col2.selectbox(t["contrast_group"], unique_conditions)
        base_group = col3.selectbox(t["base_group"], unique_conditions, index=len(unique_conditions) - 1)

        if st.button(t["run_btn"]):
            with st.spinner(t["spinner"]):
                results_df = run_deseq2(counts_df, meta_df, design_col, contrast_group, base_group)
                st.session_state['results'] = results_df
                st.success(t["success"])

    except Exception as e:
        st.error(f"{t['error']} {e}")
else:
    st.info(t["waiting"])

# --- Results & Visualization ---
if 'results' in st.session_state:
    df = st.session_state['results'].copy()


    def categorize_gene(row: pd.Series, lfc_thresh: float, padj_thresh: float) -> str:
        """Categorizes genes based on statistical significance and fold change."""
        if row['padj'] < padj_thresh and row['log2FoldChange'] > lfc_thresh:
            return f"{t['upregulated']} ({contrast_group})"
        elif row['padj'] < padj_thresh and row['log2FoldChange'] < -lfc_thresh:
            return f"{t['downregulated']} ({contrast_group})"
        return 'Not Significant'


    df['Status'] = df.apply(categorize_gene, axis=1, args=(lfc_threshold, padj_threshold))

    st.markdown(t["results_header"])
    c1, c2, c3 = st.columns(3)
    n_up = len(df[df['Status'] == f"{t['upregulated']} ({contrast_group})"])
    n_down = len(df[df['Status'] == f"{t['downregulated']} ({contrast_group})"])

    c1.metric(t["total_genes"], len(df))
    c2.metric(t["upregulated"], n_up)
    c3.metric(t["downregulated"], n_down)

    color_map = {
        f"{t['upregulated']} ({contrast_group})": '#ef4444',
        f"{t['downregulated']} ({contrast_group})": '#3b82f6',
        'Not Significant': '#e5e7eb'
    }

    fig = px.scatter(
        df, x='log2FoldChange', y='-log10(padj)', color='Status', color_discrete_map=color_map,
        hover_name='Gene', hover_data={'Status': False, '-log10(padj)': ':.2f', 'log2FoldChange': ':.2f'},
        height=600
    )

    fig.add_hline(y=-np.log10(padj_threshold), line_dash="dash", line_color="grey")
    fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="grey")
    fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="grey")
    fig.update_layout(plot_bgcolor='white')

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label=t["download"],
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='deseq2_results.csv',
        mime='text/csv',
    )