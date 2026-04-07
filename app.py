import streamlit as st 
import pandas as pd 
import numpy as np
import io
import seaborn as sns
import pandas.api.types as pd_types
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# AutoML Explorer Streamlit app.
# Workflow: upload -> clean -> encode -> train -> evaluate -> download.

# Configure page layout and title.
st.set_page_config(page_title="AutoML Explorer", layout="wide", initial_sidebar_state="expanded")

with open("style2.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Small helper to render consistent visual dividers.
def divider(spacing="normal"):
    """Create a styled divider with optional spacing"""
    if spacing == "large":
        st.markdown("<hr style='border: 2px solid #1fd2db; margin: 30px 0;'>", unsafe_allow_html=True)
    elif spacing == "small":
        st.markdown("<hr style='border: 1px solid #1fd2db; margin: 10px 0;'>", unsafe_allow_html=True)
    else:  # normal
        st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)

def alert(msg, kind="success"):
    colors = {
        "success": ("rgba(16,185,129,0.4)", "var(--success)", "rgba(16,185,129,0.5)"),
        "warning":    ("rgba(245,158,11,0.4)",  "var(--warning)",    "rgba(245,158,11,0.5)"),
        "error":  ("rgba(239,68,68,0.4)",   "var(--error)",  "rgba(239,68,68,0.5)"),
        "info":    ("rgba(0,212,200,0.4)",   "var(--accent)",  "rgba(0,212,200,0.5)"),
    }
    bg, color, border = colors.get(kind, colors["info"])
    st.markdown(f"""
        <div style='background:{bg}; border-left:4px solid {border}; 
                    border-radius:6px; padding:12px 16px; margin:8px 0;
                    font-family:var(--sans); font-size:0.90rem; color:var(--text); line-height:1.5'>
            {msg}
        </div>""", unsafe_allow_html=True)


st.markdown("""
    <div style='padding: 60px 0 40px 0; align-items:center; text-align:center; display:flex; flex-direction:column; justify-content:center; max-width:900px; margin:auto; position:relative'>
        <p class='section-label'>Automated Machine Learning</p>
        <h1 class='page-title'>AutoML Explorer</h1> 
        <p class='page-subtitle' style='margin-top:22px; max-width:520px; align-items:center; text-align:center'>
            Upload any tabular dataset. Get a trained, evaluated, 
            and downloadable model in minutes.
        </p>
    </div>
""", unsafe_allow_html=True)




col1, col2, col3 = st.columns(3)

col1.markdown("""
    <div class='how-card'>
        <p class='how-number'>Step 01</p>
        <p class='how-title'>Upload Dataset</p>
        <p class='how-desc'>CSV or Excel - we detect column types, duplicates, and missing values automatically.</p>
    </div>""", unsafe_allow_html=True)

col2.markdown("""
    <div class='how-card'>
        <p class='how-number'>Step 02</p>
        <p class='how-title'>Clean & Encode</p>
        <p class='how-desc'>Handle missing values, encode categorical features, and set your target column.</p>
    </div>""", unsafe_allow_html=True)

col3.markdown("""
    <div class='how-card'>
        <p class='how-number'>Step 03</p>
        <p class='how-title'>Train & Export</p>
        <p class='how-desc'>We compare every available model, tune the best one automatically, and give you a ready to use model file.</p>
    </div>""", unsafe_allow_html=True)

divider("large")

file_upload = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])




if file_upload is not None:

    if st.session_state.get("uploaded_filename") != file_upload.name:
        if file_upload.name.endswith(".csv"):
            st.session_state.df_original = pd.read_csv(file_upload)
        elif file_upload.name.endswith((".xlsx", ".xls")):
            st.session_state.df_original = pd.read_excel(file_upload)
        st.session_state.uploaded_filename = file_upload.name

        for key in [
            "best_model", "results", "table", "best_config",
            "select_target", "target_label_encoder", "original_target_labels",
            "trained_as_classification"
        ]:
            st.session_state.pop(key, None)


    st.markdown("<div class='box'>", unsafe_allow_html=True)
    ext = file_upload.name.split(".")[-1].upper()
    alert(f"Your {ext} file was uploaded successfully", "success")
    st.markdown("<h4 class='title'>Dataset Preview</h4>", unsafe_allow_html=True)
    st.dataframe(st.session_state.df_original)
    st.markdown("</div>", unsafe_allow_html=True)


    # Check duplicate rows.
    divider("small")

    st.markdown("""
        <p class='section-label'>Data Inspection</p>
        <p class='section-title'>Duplicate Rows</p>
    """, unsafe_allow_html=True)

    dup_count = st.session_state.df_original.duplicated().sum()

    if dup_count > 0:

        alert(f"{dup_count} duplicate rows detected", "warning")

        if st.button("Remove Duplicates"):
            st.session_state.df_original = st.session_state.df_original.drop_duplicates().reset_index(drop=True)
            alert(f"Removed {dup_count} duplicate rows. New shape: {st.session_state.df_original.shape[0]} rows", "success")   

    else:
        alert("No duplicate rows detected", "success")


    divider()

    # Column type review.
    st.markdown("""
        <p class='section-label'>Data Inspection</p>
        <p class='section-title'>Column Types</p>
        <p class='section-desc'>Review and fix column types before proceeding. Wrong types cause encoding and training issues.</p>
    """, unsafe_allow_html=True)

    # Show current dtypes in a summary table.
    dtype_df = pd.DataFrame({
        "Column": st.session_state.df_original.columns,
        "Current Type": [str(st.session_state.df_original[col].dtype) for col in st.session_state.df_original.columns],
        "Sample Values": [str(st.session_state.df_original[col].dropna().iloc[0]) if len(st.session_state.df_original[col].dropna()) > 0 else "N/A" for col in st.session_state.df_original.columns],
        "Unique Values": [st.session_state.df_original[col].nunique() for col in st.session_state.df_original.columns],
        "Missing": [st.session_state.df_original[col].isnull().sum() for col in st.session_state.df_original.columns]
    })

    styled_dtype = dtype_df.style\
        .set_properties(**{
            "color": "white",
            "font-size": "13px",
            "text-align": "center",
            "border": "1px solid #1fd2db",
            "background-color": "#262730"
        })\
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#1fd2db"),
                ("color", "black"),
                ("font-size", "13px"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("padding", "8px")
            ]
        }])

    st.dataframe(styled_dtype, use_container_width=True)

    divider("small")

    # Manual column type conversion.
    st.markdown("""
        <p class='section-label'>Type Conversion</p>
        <p class='section-title'>Convert Column</p>
        <p class='section-desc'>If a column has the wrong type, convert it here before cleaning and encoding.</p>
    """, unsafe_allow_html=True)


    col_to_convert = st.selectbox(
        "Select column to convert",
        st.session_state.df_original.columns.tolist(),
        key="col_to_convert"
    )

    current_dtype = str(st.session_state.df_original[col_to_convert].dtype)

    st.markdown(f"<p style='color:#1fd2db; font-size:13px'>Current type: <span style='color:white; font-weight:600'>{current_dtype}</span></p>", unsafe_allow_html=True)

    target_dtype = st.selectbox(
        "Convert to",
        ["-- Select type --", "Text (object)", "Numeric (float64)"],
        key="target_dtype"
    )

    if st.button("Convert Type"):
        if target_dtype == "-- Select type --":
            alert("Please select a target type.", "warning")
        else:
            try:
                if target_dtype == "Text (object)":
                    st.session_state.df_original[col_to_convert] = st.session_state.df_original[col_to_convert].astype("object")
                elif target_dtype == "Numeric (float64)":
                    st.session_state.df_original[col_to_convert] = pd.to_numeric(st.session_state.df_original[col_to_convert], errors="coerce")
                
                alert(f"'{col_to_convert}' converted to {target_dtype} successfully!", "success")

                # Refresh local working dataframe.
                df = st.session_state.df_original.copy()

                # Show updated dtype table.
                updated_dtype_df = pd.DataFrame({
                    "Column": st.session_state.df_original.columns,
                    "Type": [str(st.session_state.df_original[col].dtype) for col in st.session_state.df_original.columns],
                })
                st.dataframe(updated_dtype_df, use_container_width=True)

            except Exception as e:
                alert(f"Conversion failed: {e}", "error")

    st.markdown("</div>", unsafe_allow_html=True)
    divider()

    # Target column selection.
    st.markdown("""
        <p class='section-label'>Step 01</p>
        <p class='section-title'>Select Target Column</p>
        <p class='section-desc'>The target is the column your model will learn to predict.</p>
    """, unsafe_allow_html=True)

    # Initialize default target only once.
    if "select_target" not in st.session_state:
        st.session_state.select_target = st.session_state.df_original.columns[0]

    # Keep selected target stable across reruns.
    available_cols = st.session_state.df_original.columns.tolist()
    saved_index = available_cols.index(st.session_state.select_target) \
                  if st.session_state.select_target in available_cols else 0

    select_target = st.selectbox("", available_cols, index=saved_index, key="select_target_widget")
    
    # If target changes, clear model-related state.
    if select_target != st.session_state.get("select_target"):
        for key in ["best_model", "results", "table", "target_label_encoder", "original_target_labels"]:
            st.session_state.pop(key, None)
    
    st.session_state.select_target = select_target

    if st.session_state.df_original[select_target].isnull().sum() == 0:
        alert(f"Target column <strong>{select_target}</strong> has no missing values", "success")
    elif st.session_state.df_original[select_target].isnull().sum() > 0:
        alert(f"Target column <strong>{select_target}</strong> has missing values", "warning")

    divider()

    # Optional feature removal.
    st.markdown("""
        <p class='section-label'>Step 02</p>
        <p class='section-title'>Remove Irrelevant Columns</p>
        <p class='section-desc'>Remove ID columns, names, or any columns that should not influence the prediction.</p>
    """, unsafe_allow_html=True)

    deletable_cols = [c for c in st.session_state.df_original.columns.tolist() if c != select_target]
    select_del = st.multiselect("", deletable_cols)
    df = st.session_state.df_original.drop(columns=select_del).copy()

    with st.sidebar:
        st.markdown("""
            <div style='padding: 20px 0 24px 0; align-items:center; text-align:center; display:flex; flex-direction:column; justify-content:center; max-width:300px; margin:auto; position:relative'>
                <p style='font-family:"DM Mono",monospace; font-size:1.9rem; 
                          letter-spacing:0.18em; font-weight:800;
                          color:#00d4c8; margin:0 0 6px 0'>AutoML</p>
                <p style='font-family:"Syne",sans-serif; font-size:1.4rem; 
                          font-weight:800; color:#e8eaf0; margin:0;  
                          letter-spacing:-0.01em;'>Explorer</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:none; border-top:1px solid #1e2330; margin:0 0 20px 0'>", unsafe_allow_html=True)

        st.markdown("<p style='font-family:\"DM Mono\",monospace; font-size:1.1rem; letter-spacing:0.15em; text-transform:uppercase; text-align:center; font-weight:700; color:#00d4c8; margin:10px 0 14px 0'>Dataset</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object']).columns)

        
        col1, col2 = st.columns(2)
        col1.markdown(f"""
            <div style='background:#262730 !important; padding:25px; border-radius:12px; 
                border-top: 3px solid #1fd2db; text-align:center; height:auto; margin-bottom:20px;
                box-shadow: 0 4px 15px rgba(31, 210, 219, 0.1)'>     
                <p style='font-family:"DM Mono",monospace; font-size:0.78rem; 
                        letter-spacing:0.1em; text-transform:uppercase;
                            color:#00d4c8; margin:0 0 4px 0'>Rows</p>
                <p style='font-family:"Syne",sans-serif; font-size:1.2rem; 
                        font-weight:700; color:#e8eaf0; margin:0'>{df.shape[0]:,}</p>
            </div>""", unsafe_allow_html=True)
        col2.markdown(f"""
            <div style='background:#262730 !important; padding:25px; border-radius:12px; 
                border-top: 3px solid #1fd2db; text-align:center; height:auto;
                box-shadow: 0 4px 15px rgba(31, 210, 219, 0.1)'>
                <p style='font-family:"DM Mono",monospace; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:#00d4c8; margin:0 0 4px 0'>Columns</p>
                <p style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700; color:#e8eaf0; margin:0'>{df.shape[1]}</p>
            </div>""", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        col3.markdown(f"""
            <div style='background:#262730 !important; padding:25px; border-radius:12px; 
                border-top: 3px solid #1fd2db; text-align:center; height:auto;
                box-shadow: 0 4px 15px rgba(31, 210, 219, 0.1)'>
                <p style='font-family:"DM Mono",monospace; font-size:0.78rem; 
                          letter-spacing:0.1em; text-transform:uppercase; 
                          color:#00d4c8; margin:0 0 4px 0'>Numeric</p>
                <p style='font-family:"Syne",sans-serif; font-size:1.2rem; 
                          font-weight:700; color:#e8eaf0; margin:0'>{num_cols}</p>
            </div>""", unsafe_allow_html=True)
        col4.markdown(f"""
            <div style='background:#262730 !important; padding:25px; border-radius:12px; 
                border-top: 3px solid #1fd2db; text-align:center; height:auto;
                box-shadow: 0 4px 15px rgba(31, 210, 219, 0.1)'>
                <p style='font-family:"DM Mono",monospace; font-size:0.78rem; 
                          letter-spacing:0.1em; text-transform:uppercase; 
                          color:#00d4c8; margin:0 0 4px 0'>Text</p>
                <p style='font-family:"Syne",sans-serif; font-size:1.2rem; 
                          font-weight:700; color:#e8eaf0; margin:0'>{cat_cols}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        missing = df.isnull().sum().sum()
        if missing == 0:
            alert("No missing values detected", "success")

        else:
            st.markdown(f"""
                <div style='background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2); 
                            border-radius:5px; padding:10px 12px'>
                    <p style='font-family:"DM Mono",monospace; font-size:0.65rem; 
                              letter-spacing:0.08em; color:#f59e0b; margin:0'>
                        {missing} missing values
                    </p>
                </div>""", unsafe_allow_html=True)
            alert(f"{missing} missing values detected", "warning")

        st.markdown("<hr style='border:none; border-top:1px solid #1e2330; margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:\"DM Mono\",monospace; font-size:1.1rem; letter-spacing:0.15em; text-transform:uppercase; text-align:center; color:#00d4c8; font-weight:700; margin:0 0 14px 0'>Target</p>", unsafe_allow_html=True)


        y_check = df[select_target]
        n_unique_check = y_check.nunique()
        unique_ratio_check = round(n_unique_check / len(y_check) * 100, 1)
        task = "Classification" if (
            pd_types.is_object_dtype(y_check) or
            pd_types.is_bool_dtype(y_check) or
            (n_unique_check <= 15 and unique_ratio_check < 5)
        ) else "Regression"

        st.markdown(f"""
            <div style='background-color:#08090d !important; padding:15px; border-radius:10px; border-left: 4px solid #00d4c8;
                box-shadow: 0 4px 15px rgba(31, 210, 219, 0.1)'>
                <p style='color:gray; margin:0; font-size:11px'>Target Column</p>
                <p style='color:white; margin:4px 0; font-size:15px; font-weight:600'>{select_target}</p>
                <p style='color:gray; margin:0; font-size:11px'>Detected Task</p>
                <p style='color:#00d4c8; margin:4px 0; font-size:15px; font-weight:600'>{task}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border:none; border-top:1px solid #1e2330; margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("""
            <p style='font-family:"DM Mono",monospace; font-size:0.6rem; 
                      letter-spacing:0.08em; color:#5a6070; text-align:center; margin:0'>
                Refreshing resets progress
            </p>""", unsafe_allow_html=True)

    
    divider("large")
    st.markdown("</div>", unsafe_allow_html=True)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    str_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # Missing value handling (categorical).
    st.markdown("""
        <p class='section-label'>Step 03 — Missing Values</p>
        <p class='section-title'>Categorical Features</p>
        <p class='section-desc'>Choose how to fill missing values in text columns.</p>
    """, unsafe_allow_html=True)

    nulls_categorical = df[str_columns].isnull().sum() if str_columns else pd.Series(dtype=int)

    if str_columns:
        if nulls_categorical.sum() == 0:
            alert("No missing values in categorical columns", "success")

        elif nulls_categorical.sum() > 0:

            if select_target in str_columns:
                alert(f"Your target column '{select_target}' has missing values. Handle this carefully - imputing target values can affect model quality.", "error")

            st.markdown("<p style='color:#1fd2db; font-weight:600'>Missing values per column:</p>", unsafe_allow_html=True)
            st.dataframe(
                nulls_categorical[nulls_categorical > 0].reset_index()
                .rename(columns={"index": "Column", 0: "Missing Count"}),
                use_container_width=True
            )

            st.markdown("<p style='color:gray; font-size:13px; margin-top:10px'>Choose a method for each column individually:</p>", unsafe_allow_html=True)

            # Show only columns that actually have missing values.
            cat_cols_with_nulls = [col for col in str_columns if df[col].isnull().sum() > 0]

            cat_col_methods = {}
            for col in cat_cols_with_nulls:
                missing_count = df[col].isnull().sum()
                st.markdown(f"""
                    <div style='background:#262730; padding:12px 15px; border-radius:8px; 
                                border-left: 3px solid #1fd2db; margin-bottom:8px'>
                        <p style='color:white; margin:0; font-size:14px; font-weight:600'>
                            {col}
                            <span style='color:gray; font-size:12px; font-weight:400'>
                                - {missing_count} missing values
                            </span>
                        </p>
                    </div>""", unsafe_allow_html=True)

                cat_col_methods[col] = st.selectbox(
                    f"Method for `{col}`",
                    ["-- Select method --", "Mode", "Fill with 'Unknown'", "Drop rows"],
                    key=f"cat_method_{col}",
                    index=0
                )

            if st.button("Apply Changes"):
                all_selected = all(v != "-- Select method --" for v in cat_col_methods.values())

                if not all_selected:
                    alert("Please select a method for every column before applying.", "warning")

                else:
                    drop_cols = []
                    for col, method in cat_col_methods.items():
                        if method == "Mode":
                            df[col] = df[col].fillna(df[col].mode()[0])
                        elif method == "Fill with 'Unknown'":
                            df[col] = df[col].fillna("Unknown")
                        elif method == "Drop rows":
                            drop_cols.append(col)

                    if drop_cols:
                        df = df.dropna(subset=drop_cols)

                    st.session_state.df_original = df.copy()
                    alert("Categorical imputation applied successfully!", "success")

            st.dataframe(df)

            remaining = df[str_columns].isnull().sum().sum()
            if remaining == 0:
                alert("No more missing values in categorical columns", "success")
            else:
                alert(f"Still {remaining} missing values remaining in categorical columns", "warning")

    else:
        alert("No categorical columns to handle", "info")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    divider()
    
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    nulls_continuous = df[numeric_columns].isnull().sum() if numeric_columns else pd.Series(dtype=int)

    # Missing value handling (numeric).
    st.markdown("""
        <p class='section-label'>Step 03 — Missing Values</p>
        <p class='section-title'>Numerical Features</p>
        <p class='section-desc'>Choose how to fill missing values in numeric columns.</p>
    """, unsafe_allow_html=True)

    if numeric_columns and nulls_continuous.sum() > 0:

        st.markdown("<p style='color:#1fd2db; font-weight:600'>Missing values per column:</p>", unsafe_allow_html=True)
        st.dataframe(
            nulls_continuous[nulls_continuous > 0].reset_index()
            .rename(columns={"index": "Column", 0: "Missing Count"}),
            use_container_width=True
        )

        st.markdown(
            "<p style='color:gray; font-size:13px; margin-top:10px'>Choose a method for each numeric column individually:</p>",
            unsafe_allow_html=True
        )

        cols_with_nulls = [col for col in numeric_columns if df[col].isnull().sum() > 0]

        col_methods = {}

        for col in cols_with_nulls:
            missing_count = df[col].isnull().sum()
            is_integer_col = pd.api.types.is_integer_dtype(df[col])

            st.markdown(f"""
                <div style='background:#262730; padding:12px 15px; border-radius:8px;
                            border-left: 3px solid #1fd2db; margin-bottom:8px'>
                    <p style='color:white; margin:0; font-size:14px; font-weight:600'>
                        {col}
                        <span style='color:gray; font-size:12px; font-weight:400'>
                            - {missing_count} missing values
                        </span>
                    </p>
                    <p style='color:#bbbbbb; margin:6px 0 0 0; font-size:12px'>
                        Detected type: {"Integer" if is_integer_col else "Float / Numeric"}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            if is_integer_col:
                method_options = ["-- Select method --", "Median", "Mode", "Drop rows", "Mean (round to int)"]
            else:
                method_options = ["-- Select method --", "Mean", "Median", "Mode", "Drop rows"]

            col_methods[col] = st.selectbox(
                f"Method for `{col}`",
                method_options,
                key=f"num_method_{col}",
                index=0
            )

            if is_integer_col:
                st.caption(f"Tip: `{col}` is integer-type, so Mean is only allowed as rounded mean.")

        if st.button("Apply Changes"):
            all_selected = all(v != "-- Select method --" for v in col_methods.values())

            if not all_selected:
                st.warning("Please select a method for every column before applying.")
            else:
                drop_cols = []

                for col, method in col_methods.items():
                    is_integer_col = pd.api.types.is_integer_dtype(df[col])

                    if method == "Mean":
                        df[col] = df[col].fillna(df[col].mean())

                    elif method == "Mean (round to int)":
                        rounded_mean = int(round(df[col].dropna().mean()))
                        df[col] = df[col].fillna(rounded_mean).astype("Int64")

                    elif method == "Median":
                        median_value = df[col].median()

                        if is_integer_col:
                            median_value = int(round(median_value))
                            df[col] = df[col].fillna(median_value).astype("Int64")
                        else:
                            df[col] = df[col].fillna(median_value)

                    elif method == "Mode":
                        mode_value = df[col].mode()[0]

                        if is_integer_col:
                            mode_value = int(mode_value)
                            df[col] = df[col].fillna(mode_value).astype("Int64")
                        else:
                            df[col] = df[col].fillna(mode_value)

                    elif method == "Drop rows":
                        drop_cols.append(col)

                if drop_cols:
                    df = df.dropna(subset=drop_cols)

                st.session_state.df_original = df.copy()
                alert("Numeric imputation applied successfully!", "success")

        st.dataframe(df)

        remaining = df[numeric_columns].isnull().sum().sum()
        if remaining == 0:
            alert("No more missing values in numerical columns", "success")
        else:
            alert(f"Still {remaining} missing values remaining in numerical columns", "warning")

    else:
        alert("No missing values detected in numeric columns", "success")

    st.markdown("</div>", unsafe_allow_html=True)

    
    st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)

    # Preview cleaned dataset.
    st.markdown("""
        <p class='section-label'>Preview</p>
        <p class='section-title'>Cleaned Dataset</p>
    """, unsafe_allow_html=True)
    st.dataframe(df)

    divider("large")

    # Quick EDA section.
    st.markdown("""
        <p class='section-label'>Step 04</p>
        <p class='section-title'>Exploratory Data Analysis</p>
    """, unsafe_allow_html=True)

    st.markdown("<h5 class='title'>Summary statistics</h5>", unsafe_allow_html=True)

    numeric_summary = df.describe().round(4)
    styled_numeric = numeric_summary.style\
        .set_properties(**{
            "color": "white",
            "font-size": "14px",
            "text-align": "center",
            "border": "1px solid #1fd2db",
            "background-color": "#262730"
        })\
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1fd2db"),
                    ("color", "black"),
                    ("font-size", "14px"),
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("padding", "10px")
                ]
            }
        ])
    
    st.dataframe(styled_numeric, use_container_width=True)

    non_numeric = df.select_dtypes(include=['object', 'bool', 'string'])
    if not non_numeric.empty:
        st.markdown("<h6 style='color:#1fd2db; margin-top:15px'>Categorical Features Summary</h6>", unsafe_allow_html=True)
        cat_summary = non_numeric.describe()
        styled_cat = cat_summary.style\
            .set_properties(**{
                "color": "white",
                "font-size": "14px",
                "text-align": "center",
                "border": "1px solid #1fd2db",
                "background-color": "#030718"
            })\
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", "#1fd2db"),
                    ("color", "black"),
                    ("font-size", "14px"),
                    ("text-align", "center"),
                    ("font-weight", "bold")
                ]
            }])
        st.dataframe(styled_cat, use_container_width=True)
    
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    str_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    divider()

    st.markdown("<h5 class='title'>Distribution of Categorical Features</h5>", unsafe_allow_html=True)
    with st.expander("Bar graph for categorical columns", expanded=True):
        if str_columns:
            selected_column = st.selectbox("Select a categorical column for bar graph", str_columns)
            st.bar_chart(df[selected_column].value_counts())
        else:
            st.warning("No categorical columns available for bar graph")

        divider()

        st.markdown("<h5 class='title'>Bar Graphs for All Categorical Columns</h5>", unsafe_allow_html=True)

        if str_columns:
            num_columns = len(str_columns)
            cols = st.columns(num_columns)
            for i, column in enumerate(str_columns):
                with cols[i]:
                    st.subheader(f"Bar graph for {column}")
                    st.bar_chart(df[column].value_counts())
        else:
            st.warning("No categorical columns available for bar graphs")

    
    st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)


    st.markdown("<h5 class='title'>Feature Relationships</h5>", unsafe_allow_html=True)
    
    with st.expander("Scatter graph for numeric columns", expanded=False):
        if numeric_columns:
            if len(numeric_columns) < 2:
                st.warning("Not enough numeric columns for scatter graph")
            elif len(numeric_columns) >= 2:
                selected_column_x = st.selectbox("Select X", numeric_columns)
                selected_column_y = st.selectbox("Select Y", numeric_columns)
                st.scatter_chart(df, x=selected_column_x, y=selected_column_y)
        else:
            st.warning("No numeric columns available for scatter plot.")


    divider()

    st.markdown("<h5 class='title'>Correlation heatmap</h5>", unsafe_allow_html=True)
    with st.expander("Correlation heatmap", expanded=False):
        
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#030718")
            fig.patch.set_facecolor("#030718")
            ax.set_facecolor("#262730")

            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 9, "color": "white"}, xticklabels=True, yticklabels=True
            )

            ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", color="white", pad=20)
            ax.tick_params(colors="white", labelsize=10)
            plt.xticks(rotation=45, ha="right", color="white", fontsize=10)
            plt.yticks(color="white", fontsize=10, va="center", ha="right", rotation=0)
            
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

    divider("large")

    # Categorical encoding.
    st.markdown("""
        <p class='section-label'>Step 05</p>
        <p class='section-title'>Encode Categorical Features</p>
        <p class='section-desc'>Convert text columns to numbers so models can process them.</p>
    """, unsafe_allow_html=True)

    str_cols_to_encode = [
        c for c in df.select_dtypes(include=['object']).columns.tolist()
    ]

    if str_cols_to_encode:
        st.markdown(f"""
            <p style='color:gray; font-size:14px'>
                Found <span style='color:#1fd2db; font-weight:600'>{len(str_cols_to_encode)}</span> 
                categorical column(s) to encode: 
                <span style='color:white'>{', '.join(str_cols_to_encode)}</span>
            </p>""", unsafe_allow_html=True)

        st.markdown("<p style='color:gray; font-size:13px; margin-top:10px'>Choose encoding method for each column individually:</p>", unsafe_allow_html=True)

        encode_methods = {}
        for col in str_cols_to_encode:
            unique_count = df[col].nunique()
            st.markdown(f"""
                <div style='background:#262730; padding:12px 15px; border-radius:8px; 
                            border-left: 3px solid #1fd2db; margin-bottom:8px'>
                    <p style='color:white; margin:0; font-size:14px; font-weight:600'>
                        {col}
                        <span style='color:gray; font-size:12px; font-weight:400'>
                            - {unique_count} unique values
                        </span>
                    </p>
                </div>""", unsafe_allow_html=True)

            encode_methods[col] = st.selectbox(
                f"Method for `{col}`",
                ["-- Select method --", "Label Encoding", "One-Hot Encoding"],
                key=f"encode_{col}",
                index=0,
                help="Label Encoding: best for ordered categories (low/medium/high). One-Hot: best for unordered categories (city, color, gender)."
            )

            if encode_methods[col] == "Label Encoding":
                unique_vals = df[col].dropna().unique().tolist()
                st.markdown(f"""
                    <p style='color:gray; font-size:12px; margin-top:5px'>
                        Assign a number to each category in 
                        <span style='color:#1fd2db'>{col}</span> 
                        (0 = lowest/negative, higher = more positive/important):
                        
                    </p>
                    """, unsafe_allow_html=True)

                mapping_cols = st.columns(min(len(unique_vals), 4))
                custom_map = {}
                used_numbers = []
                mapping_valid = True

                for j, val in enumerate(unique_vals):
                    with mapping_cols[j % 4]:
                        num = st.number_input(
                            f"`{val}`",
                            min_value=0,
                            max_value=len(unique_vals) - 1,
                            value=j,
                            step=1,
                            key=f"map_{col}_{val}",
                        )
                        custom_map[val] = int(num)
                        used_numbers.append(int(num))

                # Validate that each category gets a unique number.
                if len(used_numbers) != len(set(used_numbers)):
                    st.warning(f"Duplicate numbers detected in `{col}` - each category must have a unique number.")
                    mapping_valid = False
                else:
                    st.markdown(f"""
                        <div style='background:#181c23; padding:8px 12px; border-radius:6px; margin-top:5px; margin-bottom:15px'>
                            <p style='color:#00d4c8; margin:0; font-size:12px'>
                                Mapping: {' → '.join([f"{k}={v}" for k, v in sorted(custom_map.items(), key=lambda x: x[1])])}
                            </p>
                        </div>""", unsafe_allow_html=True)

                # Store mapping in session state for Apply button.
                st.session_state[f"custom_map_{col}"] = custom_map
                st.session_state[f"mapping_valid_{col}"] = mapping_valid

        if st.button("Apply Encoding"):
            all_selected = all(v != "-- Select method --" for v in encode_methods.values())

            if not all_selected:
                st.warning("Please select a method for every column before applying.")
            else:
                from sklearn.preprocessing import LabelEncoder

                # Save original target labels before encoding.
                st.session_state.original_target_labels = (
                    df[select_target].dropna().unique().tolist()
                )

                ohe_cols = []
                le = LabelEncoder()

                # Apply selected encoding method per column.
                for col, method in encode_methods.items():
                    if method == "Label Encoding":
                        custom_map = st.session_state.get(f"custom_map_{col}")
                        mapping_valid = st.session_state.get(f"mapping_valid_{col}", True)

                        if custom_map and mapping_valid:
                            # Use user-defined mapping.
                            df[col] = df[col].map(custom_map)
                            # Save mapping for confusion matrix labels later.
                            if col == select_target:
                                st.session_state.target_label_encoder = {
                                    v: k for k, v in custom_map.items()
                                }
                        else:
                            # Fallback to sklearn LabelEncoder.
                            df[col] = le.fit_transform(df[col].astype(str))
                            if col == select_target:
                                st.session_state.target_label_encoder = le

                    elif method == "One-Hot Encoding":
                        ohe_cols.append(col)

                if ohe_cols:
                    df = pd.get_dummies(df, columns=ohe_cols)

                st.session_state.df_original = df.copy()
                alert(f"Encoding applied successfully! New shape: {df.shape[0]} rows × {df.shape[1]} columns", "success")
                st.rerun()
                st.dataframe(df)

    else:
        
        alert("No categorical columns to encode - data is ready for training!", "success")


    divider("large")
    st.markdown("</div>", unsafe_allow_html=True)

    # Data quality checks before training.
    st.markdown("""
        <p class='section-label'>Pre-flight Check</p>
        <p class='section-title'>Data Quality</p>
        <p class='section-desc'>Verifying your data is ready for training.</p>
    """, unsafe_allow_html=True)

    issues = []

    if df.isnull().sum().sum() > 0:
        issues.append(f"Still has {df.isnull().sum().sum()} missing values - handle them first")

    str_remaining = [c for c in df.columns if c != select_target 
                     and not pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_bool_dtype(df[c])]
    if str_remaining:
        issues.append(f"Unencoded columns: {', '.join(str_remaining)} - encode them first")
        alert(f"Unencoded columns detected: {', '.join(str_remaining)}", "error")
        alert(f"Unencoded columns detected: {', '.join(str_remaining)}", "warning")
        

    if len(df) < 50:
        issues.append(f"Only {len(df)} rows - too few for reliable training")

    if df[select_target].nunique() == 1:
        issues.append("Target column has only 1 unique value - cannot train")

    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()
    else:
        alert("Data quality check passed - ready for training!", "success")

    divider()

    y = df[select_target]
    n_unique = y.nunique()
    unique_ratio = round(n_unique / len(y) * 100, 1)

    suggested = "Classification" if (pd_types.is_object_dtype(y) or pd_types.is_bool_dtype(y) or (n_unique <= 15 and unique_ratio < 5)) else "Regression"

    # Task type selection.
    st.markdown("""
        <p class='section-label'>Step 06</p>
        <p class='section-title'>Select Task Type</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='background:#262730; padding:15px; border-radius:10px; 
                    border-left: 4px solid #1fd2db; margin-bottom:15px'>
            <p style='color:gray; margin:0; font-size:12px'>Target Column Analysis - <span style='color:white'>{select_target}</span></p>
            <p style='color:white; margin:8px 0 4px 0'>
                Unique values: <span style='color:#1fd2db; font-weight:600'>{n_unique}</span> 
                out of <span style='color:#1fd2db; font-weight:600'>{len(y)}</span> rows 
                (<span style='color:#1fd2db'>{unique_ratio}%</span>)
            </p>
            <p style='color:gray; margin:0; font-size:12px'>
                Suggestion: <span style='color:#1fd2db; font-weight:600'>{suggested}</span> -
                {'string/bool target or few unique values detected' if suggested == 'Classification' else 'many unique numeric values detected - likely a continuous target'}
            </p>
        </div>""", unsafe_allow_html=True)

    task_type = st.radio(
        "Confirm task type",
        ["Classification", "Regression"],
        index=0 if suggested == "Classification" else 1,
        help="Classification: predict a category (yes/no, type, label). Regression: predict a number (price, salary, temperature)."
    )
    is_classification = task_type == "Classification"

    divider()

    # Training settings.
    st.markdown("""
        <p class='section-label'>Step 07</p>
        <p class='section-title'>Training Settings</p>
        <p class='section-desc'>Adjust these to control training speed and quality.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Cross-validation folds.
        cv_folds = st.slider(
            "Cross-validation folds",
            min_value=2, max_value=5, value=3,
            help="Number of folds for cross-validation. More folds = better model evaluation but slower training. 3 is a good balance, 5 for more reliable results."
        )

        top_n_models = 5  # keep the comparison set small to reduce shared CPU usage

        

        

    with col2:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10, max_value=90, value=20,
            help="Percentage of data used for testing. 20% is standard."
        ) / 100

        

    # with col3:
    #     n_iter = st.slider(
    #         "Hyperparameter tuning iterations",
    #         min_value=10, max_value=500, value=50, step=10,
    #         help="How many hyperparameter combinations Optuna tries. More iterations = better results but slower. 50 is a good balance, 200+ for best possible results."
    #     )

    


        
    normalize = False
    normalize_method = "zscore"

    remove_outliers = False
    train_on_full = False

    # Advanced options.
    with st.expander("Advanced Settings", expanded=False):
        col3, col4 = st.columns(2)

        with col3:

            normalize = st.checkbox(
                "Normalize features",
                value=False,
                help="Scales all numeric features to same range. Helps models like Ridge, SVM, KNN. Not needed for tree-based models like XGBoost, Random Forest."
            )

            if normalize:
                normalize_method = st.selectbox(
                    "Normalization method",
                    ["zscore", "minmax", "maxabs", "robust"],
                    help="zscore: standard scaler, recommended for most models. minmax: scales to 0-1. maxabs: scales to -1 to 1. robust: best when data has outliers."
                )

            # Outlier removal (regression only).
            if not is_classification:
                st.markdown("<br>", unsafe_allow_html=True)
                remove_outliers = st.checkbox(
                    "Remove outliers from target column",
                    value=False,
                    help="Removes extreme values that hurt regression performance. Recommended if MAE/RMSE is very high."
                )

                if remove_outliers:
                    outlier_threshold = st.slider(
                        "Outlier sensitivity",
                        min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                        help="Lower = removes more outliers. 1.5 is standard (IQR method)."
                    )
                    Q1 = df[select_target].quantile(0.25)
                    Q3 = df[select_target].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - outlier_threshold * IQR
                    upper = Q3 + outlier_threshold * IQR
                    outliers_count = df[(df[select_target] < lower) | (df[select_target] > upper)].shape[0]

                    if outliers_count > 0:
                        st.warning(f"⚠️ {outliers_count} outlier rows will be removed before training.")
                        df = df[(df[select_target] >= lower) & (df[select_target] <= upper)].copy()
                        st.session_state.df_original = df.copy()
                    else:
                        alert("No outliers detected in target column.","success")

        with col4:
            train_on_full = st.checkbox(
                "Train final model on full dataset",
                value=False,
                help="ON: trains on 100% of data before download - strongest model. || OFF: trains on 80% only - safer if you want to keep test set untouched."
            )

    # st.markdown("""
    #     <div style='background:#262730; padding:15px; border-radius:10px; 
    #                 border-left: 4px solid #1fd2db; margin-top:15px; margin-bottom: 20px'>
    #         <p style='color:#1fd2db; font-size:13px; font-weight:600'>
    #             How to get the best results:
    #         </p>
    #         <ul style='color:gray; margin:8px 0 0 0; font-size:12px'>
    #             <li><span style='color:white'>Fast + No Tuning</span> - quickest, good for exploring your data</li>
    #             <li><span style='color:white'>Fast + Tuning</span> - balanced speed and quality</li>
    #             <li><span style='color:white'>Thorough + Tuning</span> - best possible results, slowest</li>
    #         </ul>
    #     </div>""", unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    with col5:

        speed_mode = st.radio(
                "Training mode",
                ["Fast", "Thorough"],
                index=0,
                help="Fast: quick screening then deep evaluation of top 5 - recommended for large datasets. Thorough: tries multiple configurations - better for small datasets under 1000 rows."
        )
    
    # with col6:
    #     use_tuning = st.checkbox(
    #         "Enable Optuna hyperparameter tuning",
    #         value=True,
    #         help="ON: Optuna finds best hyperparameters - better results but slower. || OFF: use model as-is - faster."
    #     )
    use_tuning = False  # disable tuning for now to reduce CPU usage on shared environment

    divider()

    # Use saved task type after training for consistent display.
    was_classification = st.session_state.get("trained_as_classification", is_classification)

    # Metrics shown in model cards.
    if was_classification:
        primary_metric   = "F1"
        secondary_metric = "AUC"
        tertiary_metric  = "Accuracy"
    else:
        primary_metric   = "R2"
        secondary_metric = "RMSE"
        tertiary_metric  = "MAE" 

    if is_classification and st.button("Train Classification Model"):
        class_pct = (y.value_counts() / len(y) * 100).round(1)
        is_imbalanced = class_pct.min() < 30

        best_result = None
        best_score = 0
        best_config_name = ""
        best_models_for_config = []


        # Candidate setup configs for classification.
        if speed_mode == "Thorough" and len(df) < 2000:
            configs = [
                {"name": "Default",         "fix_imbalance": is_imbalanced, "fold": cv_folds},
                {"name": "More CV folds",   "fix_imbalance": is_imbalanced, "fold": min(cv_folds + 2, 10)},
                {"name": "Imbalance fixed", "fix_imbalance": True,          "fold": cv_folds},
            ]
        else:
            configs = [
                {"name": "Default", "fix_imbalance": is_imbalanced, "fold": cv_folds},
            ]
        
        
        progress = st.progress(0)
        status = st.empty()

        try:
            from pycaret.classification import setup, compare_models, pull

            sort_metric = "F1" if is_imbalanced else "Accuracy"
            alert(f"Comparing all {top_n_models} available models ranked by {sort_metric} - {'imbalanced data detected' if is_imbalanced else 'balanced data detected'}", "info")


            for i, config in enumerate(configs):
                alert(f"Trying config: {config['name']}...", "info")
                
                progress.progress(int((i / len(configs)) * 100))

                experiment = setup(
                    df, target=select_target,
                    session_id=42,
                    fix_imbalance=config["fix_imbalance"],
                    fold=config["fold"],
                    train_size=1 - test_size,
                    normalize=normalize,           
                    normalize_method=normalize_method,  
                    verbose=False
                )

                models = compare_models(
                    verbose=False,
                    sort=sort_metric,
                    n_select=top_n_models
                )
                if not isinstance(models, list):
                    models = [models]

                results = pull()
                score = results.iloc[0][sort_metric]

                if score > best_score:
                    best_score = score
                    best_result = results
                    best_config_name = config["name"]
                    best_models_for_config = models
                    best_model = models[0]

            progress.progress(100)
            alert(f"Best config found: {best_config_name} with {sort_metric} = {round(best_score, 4)}", "success")

            # Optional hyperparameter tuning.
            # with st.spinner("Tuning best model with Optuna..."):
            #     if use_tuning:
            #         try:
            #             from pycaret.classification import tune_model
            #             tuned_model = tune_model(
            #                 best_model,
            #                 optimize=sort_metric,
            #                 n_iter=n_iter,
            #                 search_library="optuna",
            #                 search_algorithm="tpe",
            #                 verbose=False
            #             )
            #             tuned_results = pull()
            #             tuned_score = tuned_results.iloc[0][sort_metric]
            #             if tuned_score > best_score:
            #                 alert(f"Tuning improved {sort_metric}: {round(best_score, 4)} → {round(tuned_score, 4)}", "success")
            #                 best_model = tuned_model
            #             else:
            #                 alert(f"Original model was already optimal ({sort_metric}: {round(best_score, 4)})", "info")
            #         except Exception as e:
            #             alert(f"Tuning skipped: {str(e)}", "warning")
            #     else:
            #         alert("Optuna tuning skipped - enable it in Training Settings for potentially better results", "info")

            # Build top 3 from the same compare run to keep table/cards aligned.
            top3_models = best_models_for_config[:3] if best_models_for_config else [best_model]
            if top3_models:
                top3_models[0] = best_model

            st.session_state["best_model"] = best_model
            st.session_state["top3_models"] = top3_models
            st.session_state["results"] = best_result
            st.session_state["table"] = best_result
            st.session_state["best_config"] = best_config_name
            st.session_state["trained_as_classification"] = True

        except Exception as e:
            st.error(f"Training failed: {e}")


    elif not is_classification and st.button("Train Regression Model"):
        best_result = None
        best_score = -999
        best_config_name = ""
        best_models_for_config = []

        # Candidate setup configs for regression.
        if speed_mode == "Thorough" and len(df) < 2000:
            configs = [
                {"name": "Default",          "df": df.copy()},
                {"name": "Outliers removed", "df": df[
                    (df[select_target] >= df[select_target].quantile(0.25) - 1.5 * (df[select_target].quantile(0.75) - df[select_target].quantile(0.25))) &
                    (df[select_target] <= df[select_target].quantile(0.75) + 1.5 * (df[select_target].quantile(0.75) - df[select_target].quantile(0.25)))
                ].copy()},
            ]
        else:
            configs = [{"name": "Default", "df": df.copy()}]

        progress = st.progress(0)
        status = st.empty()

        try:
            from pycaret.regression import setup, compare_models, pull

            alert(f"Comparing all {top_n_models} available models ranked by R² - higher is better", "info")

            for i, config in enumerate(configs):
                alert(f"Trying config: {config['name']}...", "info")
                progress.progress(int((i / len(configs)) * 100))

                experiment = setup(
                    config["df"], target=select_target,
                    session_id=42,
                    fold=cv_folds,
                    train_size=1 - test_size,
                    normalize=normalize,           
                    normalize_method=normalize_method,  
                    verbose=False
                )

                models = compare_models(
                    verbose=False,
                    sort="R2",
                    n_select=top_n_models
                )
                if not isinstance(models, list):
                    models = [models]

                results = pull()
                score = results.iloc[0]["R2"]

                if score > best_score:
                    best_score = score
                    best_result = results
                    best_config_name = config["name"]
                    best_models_for_config = models
                    best_model = models[0]

            progress.progress(100)
            alert(f"Best config found: {best_config_name} with R² = {round(best_score, 4)}", "success")

            # Optional hyperparameter tuning.
            # with st.spinner("Tuning best model with Optuna..."):
            #     if use_tuning:
            #         try:
            #             from pycaret.regression import tune_model
            #             tuned_model = tune_model(
            #                 best_model,
            #                 optimize="R2",
            #                 n_iter=n_iter,
            #                 search_library="optuna",
            #                 search_algorithm="tpe",
            #                 verbose=False
            #             )
            #             tuned_results = pull()
            #             tuned_score = tuned_results.iloc[0]["R2"]
            #             if tuned_score > best_score:
            #                 alert(f"Tuning improved R²: {round(best_score, 4)} → {round(tuned_score, 4)}", "success")
            #                 best_model = tuned_model
            #             else:
            #                 alert(f"Original model was already optimal (R²: {round(best_score, 4)})", "info")
            #         except Exception as e:
            #             alert(f"Tuning skipped: {str(e)}", "warning")
            #     else:
            #         alert("Optuna tuning skipped - enable it in Training Settings for potentially better results", "info")

            # Build top 3 from the same compare run to keep table/cards aligned.
            top3_models = best_models_for_config[:3] if best_models_for_config else [best_model]
            if top3_models:
                top3_models[0] = best_model

            st.session_state["best_model"] = best_model
            st.session_state["top3_models"] = top3_models
            st.session_state["results"] = best_result
            st.session_state["table"] = best_result
            st.session_state["best_config"] = best_config_name
            st.session_state["trained_as_classification"] = False

        except Exception as e:
            alert(f"Training failed: {str(e)}", "error")

    # Refresh task-specific metric names after any training action.
    # This avoids stale labels (for example regression metrics on classifiers).
    was_classification = st.session_state.get("trained_as_classification", is_classification)
    if was_classification:
        primary_metric   = "F1"
        secondary_metric = "AUC"
        tertiary_metric  = "Accuracy"
    else:
        primary_metric   = "R2"
        secondary_metric = "RMSE"
        tertiary_metric  = "MAE"

    if st.session_state.get('results') is not None:
        st.markdown("<h4 style='color:#1fd2db; margin-top:20px'>Model Comparison Results</h4>", unsafe_allow_html=True)
        
        results_df = st.session_state['results']
        
        styled_results = results_df.style\
            .background_gradient(cmap="Blues")\
            .set_properties(**{
                "color": "white",
                "font-size": "14px",
                "text-align": "center",
                "border": "1px solid #1fd2db",
                "background-color": "#030718"
            })\
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1fd2db"),
                        ("color", "black"),
                        ("font-size", "14px"),
                        ("text-align", "center"),
                        ("font-weight", "bold")
                    ]
                },
                {
                    "selector": "tr:first-child",
                    "props": [
                        ("border-top", "3px solid #1fd2db")
                    ]
                }
            ])\
            .highlight_max(color="#1fd2db", subset=results_df.select_dtypes(include="number").columns)\
            .highlight_min(color="#ff4b4b", subset=results_df.select_dtypes(include="number").columns)
        
        st.dataframe(styled_results, use_container_width=True)

    # Top-3 model cards and manual selection.
    if st.session_state.get('best_model') is not None:
        top3 = st.session_state.get("top3_models", [st.session_state["best_model"]])
        results_df = st.session_state.get("results")

        # Model selection UI.
        st.markdown("""
            <p class='section-label'>Results</p>
            <p class='section-title'>Select Your Model</p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <p style='color:gray; font-size:13px'>
                We compared all models and found the top 3. 
                <span style='color:#1fd2db'> Recommended</span> is selected based on 
                <span style='color:white; font-weight:600'>{primary_metric}</span> first, then 
                <span style='color:white; font-weight:600'>{secondary_metric}</span> as tiebreaker. 
                You can switch if you prefer a different model.
            </p>""", unsafe_allow_html=True)
        # Build cards.
        cols = st.columns(len(top3))
        selected_model_idx = st.session_state.get("selected_model_idx", 0)

        # Rank by primary metric, then secondary, then tertiary.
        def get_model_scores(row_idx):
            try:
                row = results_df.iloc[row_idx]
                p = round(float(row.get(primary_metric, 0)), 4)
                s = round(float(row.get(secondary_metric, 0)), 4)
                t = round(float(row.get(tertiary_metric, 0)), 4)
                return p, s, t
            except Exception:
                return 0, 0, 0

        # Pick best card index by metric priority.
        best_idx = 0
        best_scores = get_model_scores(0)
        for idx in range(min(len(top3), len(results_df))):
            scores = get_model_scores(idx)
            if (scores[0] > best_scores[0] or
               (scores[0] == best_scores[0] and scores[1] > best_scores[1]) or
               (scores[0] == best_scores[0] and scores[1] == best_scores[1] and scores[2] > best_scores[2])):
                best_scores = scores
                best_idx = idx

        # Set default selected model once.
        if "selected_model_idx" not in st.session_state:
            st.session_state["selected_model_idx"] = best_idx
            st.session_state["best_model"] = top3[best_idx]

        selected_model_idx = st.session_state.get("selected_model_idx", best_idx)

        for i, model in enumerate(top3):
            model_name = type(model).__name__

            # Read scores by row position.
            p_score, s_score, t_score = get_model_scores(i)

            border_color = "#1fd2db" if i == selected_model_idx else "#444"

            # Mark recommended model using metric priority.
            if i == best_idx:
                badge = "Recommended"
                badge_color = "#1fd2db"
            else:
                badge = f"#{i+1}"
                badge_color = "gray"

            with cols[i]:
                st.markdown(f"""
                    <div style='background:#262730; padding:20px; border-radius:12px;
                                border: 2px solid {border_color}; text-align:center; margin-bottom:15px;
                                min-height:200px'>
                        <p style='color:{badge_color}; margin:0; font-size:15px; font-weight:600'>{badge}</p>
                        <h4 style='color:white; margin:8px 0; font-size:15px; font-weight:700'>{model_name}</h4>
                        <p style='color:gray; margin:4px 0 0 0; font-size:15px'>{primary_metric}</p>
                        <h2 style='color:#1fd2db; margin:2px 0'>{p_score}</h2>
                        <p style='color:gray; margin:4px 0 0 0; font-size:15px'>{secondary_metric}: <span style='color:white'>{s_score}</span></p>
                        <p style='color:gray; margin:2px 0 0 0; font-size:15px'>{tertiary_metric}: <span style='color:white'>{t_score}</span></p>
                    </div>""", unsafe_allow_html=True)

                if st.button(f"Select this model", key=f"select_model_{i}"):
                    st.session_state["selected_model_idx"] = i
                    st.session_state["best_model"] = top3[i]
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Show selected model details.
        selected_model = st.session_state["best_model"]
        selected_name = type(selected_model).__name__

        st.markdown(f"""
            <div class='selected-model-card'>
                <p class='label'>Selected Model</p>
                <p class='name'>{selected_name}</p>
                <p style='color:var(--muted); font-size:0.78rem; margin:0'>Selected for evaluation and download</p>
                <p class='params'>{str(selected_model)}</p>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Evaluation and export section.
    divider("large")

    # Evaluate.
    st.markdown("""
        <p class='section-label'>Step 08</p>
        <p class='section-title'>Evaluate & Export</p>
    """, unsafe_allow_html=True)

    if st.session_state.get("best_model") is None:
        alert("Please run Train first to find the best model.", "info")

    else:
        if st.button("Evaluate Model"):
            from sklearn.model_selection import train_test_split

            X = df.drop(columns=[select_target])
            y = df[select_target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            best_model = st.session_state["best_model"]

            # Finalize model (train on full data before export).
            with st.spinner("Finalizing model on full dataset..."):
                try:
                    if was_classification:
                        from pycaret.classification import finalize_model
                    else:
                        from pycaret.regression import finalize_model

                    final_model = finalize_model(best_model)
                    alert("Model finalized on 100% of data successfully!", "success")
                except Exception as e:
                    final_model = best_model
                    alert(f"Could not finalize - using tuned model: {str(e)}", "error")

            alert("Each training run is independent.\nRunning training again will replace these results. Your previous model will be overwritten.", "info")
                
            

            # Evaluate on held-out test set for metrics/charts.
            # Export still uses the finalized full-data model.
            eval_model = best_model

            # Extract sklearn estimator for prediction/evaluation.
            try:
                eval_model_sk = eval_model.steps[-1][1]
                eval_model_sk.fit(X_train, y_train)
            except Exception:
                eval_model_sk = eval_model
                try:
                    eval_model_sk.fit(X_train, y_train)
                except Exception:
                    eval_model_sk = None

            if eval_model_sk is not None:
                preds = eval_model_sk.predict(X_test)

                if was_classification:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

                    class_pct = (pd.Series(y_test).value_counts() / len(y_test) * 100).round(1)
                    is_imbalanced_eval = class_pct.min() < 30
                    avg_method = "weighted" if is_imbalanced_eval else "macro"

                    accuracy  = round(accuracy_score(y_test, preds), 4)
                    precision = round(precision_score(y_test, preds, average=avg_method, zero_division=0), 4)
                    recall    = round(recall_score(y_test, preds, average=avg_method, zero_division=0), 4)
                    f1        = round(f1_score(y_test, preds, average=avg_method, zero_division=0), 4)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(f"""
                        <div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; margin-top:15px; text-align:center'>
                            <p style='color:#1fd2db; margin:0; font-size:18px; font-weight:600'>Accuracy</p>
                            <h2 style='color:white; margin:0'>{accuracy}</h2>
                        </div>""", unsafe_allow_html=True)
                    col2.markdown(f"""
                        <div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; margin-top:15px; text-align:center'>
                            <p style='color:#1fd2db; margin:0; font-size:18px; font-weight:600'>Precision</p>
                            <h2 style='color:white; margin:0'>{precision}</h2>
                        </div>""", unsafe_allow_html=True)
                    col3.markdown(f"""
                        <div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; margin-top:15px; text-align:center'>
                            <p style='color:#1fd2db; margin:0; font-size:18px; font-weight:600'>Recall</p>
                            <h2 style='color:white; margin:0'>{recall}</h2>
                        </div>""", unsafe_allow_html=True)
                    col4.markdown(f"""
                        <div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; margin-top:15px; text-align:center'>
                            <p style='color:#1fd2db; margin:0; font-size:18px; font-weight:600'>F1 Score</p>
                            <h2 style='color:white; margin:0'>{f1}</h2>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)


                    # Quick overfitting check.
                    from sklearn.metrics import f1_score as f1_fn, accuracy_score as acc_fn
                    try:
                        train_preds    = eval_model_sk.predict(X_train)
                        train_accuracy = round(acc_fn(y_train, train_preds), 4)
                        train_f1       = round(f1_fn(y_train, train_preds, average=avg_method, zero_division=0), 4)
                        gap_accuracy   = round(train_accuracy - accuracy, 4)
                        gap_f1         = round(train_f1 - f1, 4)
                        
                        if gap_accuracy > 0.15 or gap_f1 > 0.15:
                            alert(f"Possible overfitting detected.\nTrain accuracy: {train_accuracy} vs Test accuracy: {accuracy} (gap: {gap_accuracy})\nThe model learned the training data too well and may not generalize.\nSuggestions:\n- Increase cross-validation folds to 7 or 10\n- Remove irrelevant or redundant columns\n- Try a simpler model like Logistic Regression or Decision Tree", "warning")
                        
                        elif gap_accuracy > 0.05:
                            alert(f"Slight variance between train and test.\nTrain accuracy: {train_accuracy} vs Test accuracy: {accuracy} — this is normal and acceptable.", "info")
                        
                        else:
                            alert(f"No overfitting detected.\nTrain accuracy: {train_accuracy} vs Test accuracy: {accuracy} - model generalizes well.", "success")
                    
                    except Exception:
                        pass

                    if accuracy < 0.7:
                        alert("Accuracy is low - suggestions:\n- Check class imbalance\n- Make sure all categorical columns are encoded\n- Remove irrelevant columns\n- Increase number of models to compare\n- Enable Optuna tuning in Training Settings", "warning")
                    
                    elif accuracy < 0.85:
                        alert("Accuracy is good - to improve further:\n- Enable Optuna tuning and increase iterations\n- Increase CV folds to 7 or 10\n- Switch to Thorough mode", "info")
                    
                    else:
                        alert("Excellent accuracy!", "success")

                    st.markdown("<h5 class='title' style='color:#1fd2db; margin-top:20px'>Classification Report</h5>", unsafe_allow_html=True)
                    report = classification_report(y_test, preds, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    styled = report_df.style\
                        .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])\
                        .set_properties(**{"color": "white", "font-size": "14px", "text-align": "center", "border": "1px solid #1fd2db", "background-color": "#030718"})\
                        .set_table_styles([{"selector": "th", "props": [("background-color", "#1fd2db"), ("color", "black"), ("font-size", "14px"), ("text-align", "center")]}])
                    st.dataframe(styled, use_container_width=True)

                    from sklearn.metrics import confusion_matrix
                    divider()
                    with st.expander("Confusion Matrix", expanded=True):
                        cm = confusion_matrix(y_test, preds)
                        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#030718")
                        fig.patch.set_facecolor("#030718")
                        ax.set_facecolor("#030718")
                        from matplotlib.colors import LinearSegmentedColormap
                        custom_cmap = LinearSegmentedColormap.from_list("custom", ["#030718", "#1fd2db"])
                        cm_percent = (cm / cm.sum() * 100).round(1)
                        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=custom_cmap, ax=ax,
                                    linewidths=2, linecolor="#030718",
                                    annot_kws={"size": 22, "color": "white", "weight": "bold"},
                                    cbar=True, vmin=0, vmax=100)
                        cbar = ax.collections[0].colorbar
                        cbar.ax.yaxis.set_tick_params(color="white")
                        cbar.set_ticks([0, 25, 50, 75, 100])
                        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")
                        ax.set_title("Confusion Matrix", color="white", fontsize=16, pad=15)
                        ax.set_xlabel("Predicted", color="#1fd2db", fontsize=13)
                        ax.set_ylabel("Actual", color="#1fd2db", fontsize=13)
                        ax.tick_params(colors="white", labelsize=12)

                        # Handle both LabelEncoder and custom mapping dict.
                        raw_labels = sorted(y_test.unique())
                        try:
                            le_saved = st.session_state.get("target_label_encoder")
                            if le_saved is None:
                                # Never encoded: labels are already readable.
                                class_labels = [str(c) for c in raw_labels]
                            elif isinstance(le_saved, dict):
                                # Custom mapping: {0: "cat", 1: "dog"}.
                                class_labels = [str(le_saved.get(v, v)) for v in raw_labels]
                            else:
                                # sklearn LabelEncoder object.
                                class_labels = [str(le_saved.classes_[v]) for v in raw_labels]
                        except Exception:
                            class_labels = [str(c) for c in raw_labels]

                        

                        ax.set_xticklabels(class_labels, color="white")
                        ax.set_yticklabels(class_labels, color="white", rotation=0)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                    from sklearn.metrics import roc_curve, auc
                    if len(np.unique(y_test)) == 2:
                        divider()
                        with st.expander("ROC Curve", expanded=False):
                            try:
                                y_prob = eval_model_sk.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                roc_auc = auc(fpr, tpr)
                                fig, ax = plt.subplots(figsize=(8, 5), facecolor="#030718")
                                ax.set_facecolor("#262730")
                                ax.plot(fpr, tpr, color="#1fd2db", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
                                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                                ax.set_xlabel("False Positive Rate", color="white")
                                ax.set_ylabel("True Positive Rate", color="white")
                                ax.set_title("ROC Curve", color="white", fontsize=14)
                                ax.tick_params(colors="white")
                                legend = ax.legend(fontsize=11)
                                for text in legend.get_texts():
                                    text.set_color("white")
                                fig.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"ROC curve not available for this model: {e}")

                    if hasattr(eval_model_sk, "feature_importances_"):
                        divider()
                        with st.expander("Feature Importance", expanded=True):
                            importances = pd.Series(eval_model_sk.feature_importances_, index=X.columns)
                            importances = importances.sort_values(ascending=True)
                            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#030718")
                            ax.set_facecolor("#262730")
                            ax.barh(importances.index, importances.values, color="#1fd2db")
                            ax.set_title("Feature Importance", color="white", fontsize=14)
                            ax.tick_params(colors="white")
                            ax.set_xlabel("Importance", color="white")
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)

                else:
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                    r2   = round(r2_score(y_test, preds), 4)
                    mae  = round(mean_absolute_error(y_test, preds), 4)
                    mse  = round(mean_squared_error(y_test, preds), 4)
                    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(f"<div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; text-align:center'><p style='color:#1fd2db; margin:0; font-size:22px; font-weight:600'>R²</p><h2 style='color:white; margin:0'>{r2}</h2></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; text-align:center'><p style='color:#1fd2db; margin:0; font-size:22px; font-weight:600'>MAE</p><h2 style='color:white; margin:0'>{mae}</h2></div>", unsafe_allow_html=True)
                    col3.markdown(f"<div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; text-align:center'><p style='color:#1fd2db; margin:0; font-size:22px; font-weight:600'>MSE</p><h2 style='color:white; margin:0'>{mse}</h2></div>", unsafe_allow_html=True)
                    col4.markdown(f"<div style='background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #1fd2db; text-align:center'><p style='color:#1fd2db; margin:0; font-size:22px; font-weight:600'>RMSE</p><h2 style='color:white; margin:0'>{rmse}</h2></div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Quick overfitting check.
                    try:
                        from sklearn.metrics import r2_score as r2_fn
                        train_preds_r = eval_model_sk.predict(X_train)
                        train_r2      = round(r2_fn(y_train, train_preds_r), 4)
                        gap_r2        = round(train_r2 - r2, 4)
                        
                        if gap_r2 > 0.2:
                            alert(f"Possible overfitting detected.\nTrain R²: {train_r2} vs Test R²: {r2} (gap: {gap_r2})\nThe model may not generalize well to new data.\nSuggestions:\n- Increase cross-validation folds\n- Remove irrelevant columns like ID or name columns\n- Enable outlier removal if not already done", "warning")
                        
                        elif gap_r2 > 0.08:
                            alert(f"Slight variance between train and test.\nTrain R²: {train_r2} vs Test R²: {r2} - acceptable for most datasets.", "info")
                        
                        else:
                            alert(f"No overfitting detected.\nTrain R²: {train_r2} vs Test R²: {r2} - model generalizes well.", "success")

                    except Exception:
                        pass

                    if r2 < 0.5:
                        alert("R² is low - suggestions:\n- Enable outlier removal\n- Remove irrelevant columns like ID or name columns\n- Check correct target column is selected\n- Enable Optuna tuning", "warning")
                    
                    elif r2 < 0.75:
                        alert("R² is decent - to improve further:\n- Increase Optuna iterations\n- Try removing outliers\n- Increase models to compare to 15", "info")
                    
                    else:
                        alert("Great R² score!", "success")

                    from sklearn.model_selection import learning_curve
                    try:
                        train_sizes, train_scores, val_scores = learning_curve(
                            eval_model_sk, X, y, cv=5,
                            scoring="neg_mean_squared_error", n_jobs=1,
                            train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42)
                        train_loss = -np.mean(train_scores, axis=1)
                        val_loss   = -np.mean(val_scores, axis=1)
                    except Exception:
                        train_sizes = np.linspace(0.1, 1.0, 5)
                        train_loss = val_loss = np.zeros(5)

                    with st.expander("Learning Curve", expanded=True):
                        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                        ax.set_facecolor("#262730")
                        ax.plot(train_sizes, train_loss, label="train loss", marker="o", color="#1fd2db")
                        ax.plot(train_sizes, val_loss,   label="val loss",   marker="o", color="orange")
                        ax.set_xlabel("Training set size", color="white", fontsize=12)
                        ax.set_ylabel("Loss", color="white", fontsize=12)
                        ax.set_title("Learning Curve", color="white", fontsize=14)
                        ax.tick_params(colors="white")
                        legend = ax.legend(fontsize=11)
                        for text in legend.get_texts():
                            text.set_color("white")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                    if hasattr(eval_model_sk, "feature_importances_"):
                        with st.expander("Feature Importance", expanded=False):
                            importances = pd.Series(eval_model_sk.feature_importances_, index=X.columns)
                            importances = importances.sort_values(ascending=True)
                            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#030718")
                            ax.set_facecolor("#262730")
                            ax.barh(importances.index, importances.values, color="#1fd2db")
                            ax.set_title("Feature Importance", color="white", fontsize=14)
                            ax.tick_params(colors="white")
                            ax.set_xlabel("Importance", color="white")
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)

                    with st.expander("Actual vs Predicted (scatter)", expanded=False):
                        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                        ax.set_facecolor("#262730")
                        sns.scatterplot(x=y_test, y=preds, ax=ax, color="#1fd2db")
                        ax.set_xlabel("Actual Values", color="white", fontsize=12)
                        ax.set_ylabel("Predicted Values", color="white", fontsize=12)
                        ax.set_title("Actual vs Predicted Values", color="white", fontsize=14)
                        ax.tick_params(colors="white")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                    with st.expander("Actual vs Predicted (line)", expanded=False):
                        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                        ax.set_facecolor("#262730")
                        ax.plot(y_test.values, label="Actual",    color="#1fd2db")
                        ax.plot(preds,         label="Predicted", color="orange")
                        ax.set_xlabel("Index", color="white", fontsize=12)
                        ax.set_ylabel("Values", color="white", fontsize=12)
                        ax.set_title("Actual vs Predicted Values", color="white", fontsize=14)
                        ax.tick_params(colors="white")
                        legend = ax.legend(fontsize=11)
                        for text in legend.get_texts():
                            text.set_color("white")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        
            # Step 09: sample predictions on unseen test rows.
            divider("large")

            st.markdown("""
                <p class='section-label'>Step 09</p>
                <p class='section-title'>Sample Predictions</p>
                <p class='section-desc'>See how the model performs on real unseen examples from your test set.</p>
            """, unsafe_allow_html=True)
        
            if st.session_state.get("best_model") is not None and st.session_state.get("results") is not None:
                
                X = df.drop(columns=[select_target])
                y = df[select_target]
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Take up to 15 random rows from test set.
                sample_size  = min(15, len(X_test))
                sample_idx   = X_test.sample(n=sample_size, random_state=99).index
                X_sample     = X_test.loc[sample_idx]
                y_sample     = y_test.loc[sample_idx]
                
                try:
                    eval_model = st.session_state["best_model"]
                    try:
                        eval_model_sk = eval_model.steps[-1][1]
                        eval_model_sk.fit(X_train, y_train)
                    except Exception:
                        eval_model_sk = eval_model
                        eval_model_sk.fit(X_train, y_train)
                    
                    preds_sample = eval_model_sk.predict(X_sample)
                    
                    # Build comparison table.
                    comparison           = X_sample.copy()
                    comparison["Actual"] = y_sample.values
                    comparison["Predicted"] = preds_sample
                    
                    if was_classification:
                        comparison["Correct"] = comparison["Actual"] == comparison["Predicted"]
                        correct_count = comparison["Correct"].sum()
                        total_count   = len(comparison)
                        
                        alert(f"{correct_count} of {total_count} sample predictions correct.\nThese are random rows from the unseen test set - the model never trained on these.", "success")
                    else:
                        comparison["Error"] = (comparison["Actual"] - comparison["Predicted"]).abs().round(4)
                        avg_error = comparison["Error"].mean().round(4)
                        alert(f"Average error on samples: {avg_error}.\nThese are random rows from the unseen test set - the model never trained on these.", "info")
                    
                    # Style the table.
                    def highlight_row(row):
                        if was_classification:
                            color = "rgba(79,255,176,0.08)" if row.get("Correct", False) else "rgba(255,79,107,0.08)"
                            return [f"background-color: {color}"] * len(row)
                        return [""] * len(row)
                    
                    styled_comparison = comparison.style.apply(highlight_row, axis=1)\
                        .set_properties(**{"color": "#e2e6f0", "font-size": "15px", "text-align": "center"})\
                        .set_table_styles([{"selector": "th", "props": [("background-color", "#1f2340"), ("color", "#9099b8"), ("font-size", "15px"), ("padding", "8px")]}])
                    
                    st.dataframe(styled_comparison, use_container_width=True)
                    
                    st.markdown("""
                        <div class='fb fb-info'>
                            <div class='fb-content'>
                                <p class='fb-title'>About these predictions</p>
                                <ul>
                                    <li>These rows were held out during training - the model has never seen them</li>
                                    <li>This gives you a realistic view of how the model will perform on real new data</li>
                                    <li>The downloaded model was trained on all data for maximum performance</li>
                                </ul>
                            </div>
                        </div>""", unsafe_allow_html=True)
                
                except Exception as e:
                    alert(f"Could not generate sample predictions: {str(e)}", "error")

            else:
                alert("Train a model first.\nSample predictions from the test set will appear here after training.", "info")

            import joblib
            buf = io.BytesIO()
            joblib.dump(final_model, buf)
            buf.seek(0)
            st.session_state["model_download_buf"] = buf.getvalue()
            alert("Model ready for download!", "success")


        if st.session_state.get("model_download_buf") is not None:
            divider()
            st.download_button(
                "Download Trained Model (.pkl)",
                data=st.session_state["model_download_buf"],
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )


else:
    st.markdown("""
        <div style='border: 1px dashed var(--border); border-radius:8px; 
                    padding:60px 40px; text-align:center; margin-top:40px;
                    background: var(--surface)'>
            <p style='font-family:var(--mono); font-size:0.65rem; letter-spacing:0.15em; 
                      text-transform:uppercase; color:var(--accent); margin:0 0 10px 0'>
                Ready to start
            </p>
            <p style='font-family:var(--display); font-size:1.4rem; font-weight:700; 
                      color:var(--text); margin:0 0 6px 0'>
                Upload your dataset above
            </p>
            <p style='font-size:0.8rem; color:var(--muted); margin:0'>
                Supported formats: CSV, Excel (.xlsx, .xls)
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; padding:20px 0; color:var(--muted); font-size:0.8rem; overflow-x: hidden; overflow-y: hidden; border-top:1px solid var(--border)'>
        Made with ❤️ by <a href='https://www.linkedin.com/in/ahmed-banafi-4b5034313/' target='_blank' style='color:var(--accent); font-weight:600; text-decoration:underline'>Ahmed Banafa</a>
    </div>
""", unsafe_allow_html=True)