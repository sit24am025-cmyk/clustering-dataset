import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering App",
    page_icon="normal_logo.png",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 7])
with col_logo:
    try:
        st.image("normal_logo.png", width=90)
    except Exception:
        st.write("🔵")
with col_title:
    st.title("Clustering App")
    st.markdown(
        "Upload a CSV or Excel dataset. This app groups records that share the "
        "**same Phone number**, the **same Email**, or the **same Address**."
    )

st.divider()

# ── Sidebar Settings ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
match_strategy = st.sidebar.radio(
    "Matching Strategy",
    ["Match ANY (Broad)", "Match ALL (Strict)"],
    index=0,
    help=(
        "**Broad:** Link records if ANY one field matches (Phone OR Email OR Address).\n\n"
        "**Strict:** Link records ONLY if all three fields match exactly."
    )
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Match ANY (Broad):** Best for finding all duplicates, even partial ones.\n\n"
    "**Match ALL (Strict):** Best for exact deduplication — all 3 fields must match."
)

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Step 1: Upload your CSV or Excel dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    # Read CSV or Excel automatically
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Preview (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Dataset loaded: **{len(df):,} rows** and **{len(df.columns)} columns**")

    st.divider()

    # ── Column Mapping ────────────────────────────────────────────────────────
    st.subheader("🔗 Step 2: Map Your Columns")
    columns = ["-- Select --"] + df.columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        phone_col = st.selectbox("📞 Phone Column", columns, index=0)
    with col2:
        email_col = st.selectbox("📧 Email Column", columns, index=0)
    with col3:
        address_col = st.selectbox("🏠 Address Column", columns, index=0)

    st.divider()

    # ── Run Clustering ────────────────────────────────────────────────────────
    if st.button("🚀 Step 3: Run Clustering", type="primary"):

        # Validate column selections
        selected = [c for c in [phone_col, email_col, address_col] if c != "-- Select --"]
        if len(selected) == 0:
            st.error("❌ Please select at least one column (Phone, Email, or Address) to cluster by.")
            st.stop()

        with st.spinner("🔄 Analyzing and clustering your records... please wait."):

            df_work = df.copy().reset_index(drop=True)

            # Values treated as blank / invalid — will NOT link records
            invalid_vals = {
                "", "nan", "none", "null", "n/a", "na", "-",
                ".", "0", "unknown", "nil", "not available"
            }

            # ── BROAD MATCHING (Match ANY) ────────────────────────────────────
            if match_strategy == "Match ANY (Broad)":

                G = nx.Graph()
                G.add_nodes_from(df_work.index)

                phone_matches  = set()
                email_matches  = set()
                address_matches = set()

                # --- Phone ---
                if phone_col != "-- Select --":
                    phones = (
                        df_work[phone_col]
                        .astype(str)
                        .str.replace(r"\D", "", regex=True)
                        .str.strip()
                    )
                    seen = {}
                    for idx, val in phones.items():
                        if val and val.lower() not in invalid_vals and len(val) >= 7:
                            if val in seen:
                                G.add_edge(idx, seen[val])
                                phone_matches.update([idx, seen[val]])
                            else:
                                seen[val] = idx

                # --- Email ---
                if email_col != "-- Select --":
                    emails = (
                        df_work[email_col]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    seen = {}
                    for idx, val in emails.items():
                        if val and val not in invalid_vals and "@" in val:
                            if val in seen:
                                G.add_edge(idx, seen[val])
                                email_matches.update([idx, seen[val]])
                            else:
                                seen[val] = idx

                # --- Address ---
                if address_col != "-- Select --":
                    addresses = (
                        df_work[address_col]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    seen = {}
                    for idx, val in addresses.items():
                        if val and val not in invalid_vals and len(val) > 5:
                            if val in seen:
                                G.add_edge(idx, seen[val])
                                address_matches.update([idx, seen[val]])
                            else:
                                seen[val] = idx

                # Extract connected components as clusters
                cluster_labels = np.zeros(len(df_work), dtype=int)
                for cluster_id, component in enumerate(nx.connected_components(G), start=1):
                    for node in component:
                        cluster_labels[node] = cluster_id

                df_work.insert(0, "Cluster_ID", cluster_labels)

            # ── STRICT MATCHING (Match ALL) ───────────────────────────────────
            else:
                phone_matches  = set()
                email_matches  = set()
                address_matches = set()

                # Use available columns for strict groupby
                temp_cols = {}

                if phone_col != "-- Select --":
                    df_work["_phone"] = (
                        df_work[phone_col]
                        .astype(str)
                        .str.replace(r"\D", "", regex=True)
                        .str.strip()
                        .str.lower()
                        .replace(list(invalid_vals), np.nan)
                    )
                    temp_cols["_phone"] = phone_col

                if email_col != "-- Select --":
                    df_work["_email"] = (
                        df_work[email_col]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .replace(list(invalid_vals), np.nan)
                    )
                    temp_cols["_email"] = email_col

                if address_col != "-- Select --":
                    df_work["_address"] = (
                        df_work[address_col]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .replace(list(invalid_vals), np.nan)
                    )
                    temp_cols["_address"] = address_col

                group_keys = list(temp_cols.keys())
                df_work["Cluster_ID"] = df_work.groupby(group_keys, dropna=False).ngroup() + 1

                # Rows where ALL columns are blank → give unique IDs (don't group blanks together)
                all_blank_mask = df_work[group_keys].isna().all(axis=1)
                if all_blank_mask.any():
                    max_id = df_work["Cluster_ID"].max()
                    n = all_blank_mask.sum()
                    df_work.loc[all_blank_mask, "Cluster_ID"] = range(max_id + 1, max_id + 1 + n)

                # Track match stats
                dup_mask = df_work.groupby("Cluster_ID")["Cluster_ID"].transform("count") > 1
                for idx in df_work[dup_mask].index:
                    if "_phone" in df_work.columns and pd.notna(df_work.at[idx, "_phone"]):
                        phone_matches.add(idx)
                    if "_email" in df_work.columns and pd.notna(df_work.at[idx, "_email"]):
                        email_matches.add(idx)
                    if "_address" in df_work.columns and pd.notna(df_work.at[idx, "_address"]):
                        address_matches.add(idx)

                # Drop temp columns
                df_work.drop(columns=group_keys, inplace=True)

                # Move Cluster_ID to front
                cols = ["Cluster_ID"] + [c for c in df_work.columns if c != "Cluster_ID"]
                df_work = df_work[cols]

            # ── Re-number Cluster IDs by order of first appearance ────────────
            seen_order = {}
            new_id_counter = 1
            new_ids = []
            for cid in df_work["Cluster_ID"]:
                if cid not in seen_order:
                    seen_order[cid] = new_id_counter
                    new_id_counter += 1
                new_ids.append(seen_order[cid])
            df_work["Cluster_ID"] = new_ids

            # ── Sort: by Cluster_ID, then original row order ──────────────────
            df_work["_orig_order"] = df_work.index
            df_work.sort_values(["Cluster_ID", "_orig_order"], inplace=True)
            df_work.drop(columns=["_orig_order"], inplace=True)
            df_work.reset_index(drop=True, inplace=True)

        # ── Results ───────────────────────────────────────────────────────────
        st.success("✅ Clustering complete!")
        st.divider()

        st.subheader("📊 Clustering Summary")

        cluster_sizes    = df_work["Cluster_ID"].value_counts()
        total_records    = len(df_work)
        total_clusters   = df_work["Cluster_ID"].nunique()
        duplicate_groups = (cluster_sizes > 1).sum()
        unique_records   = (cluster_sizes == 1).sum()
        total_duplicates = total_records - unique_records

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📋 Total Records",    f"{total_records:,}")
        m2.metric("🔢 Total Clusters",   f"{total_clusters:,}")
        m3.metric("⚠️ Duplicate Groups", f"{duplicate_groups:,}")
        m4.metric("🗂️ Duplicate Rows",   f"{total_duplicates:,}")
        m5.metric("✅ Unique Records",    f"{unique_records:,}")

        m6, m7, m8 = st.columns(3)
        m6.metric("📞 Matched by Phone",   f"{len(phone_matches):,}")
        m7.metric("📧 Matched by Email",   f"{len(email_matches):,}")
        m8.metric("🏠 Matched by Address", f"{len(address_matches):,}")

        st.info(
            f"📌 **{total_records:,}** total records → "
            f"**{total_clusters:,}** clusters → "
            f"**{duplicate_groups:,}** duplicate groups → "
            f"**{unique_records:,}** fully unique records (no match found)."
        )

        st.divider()
        st.subheader("📄 Clustered Dataset")
        st.dataframe(df_work, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        csv_data = df_work.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Step 4: Download Clustered CSV",
            data=csv_data,
            file_name="clustered_output.csv",
            mime="text/csv",
            type="primary"
        )

else:
    # ── Welcome placeholder ───────────────────────────────────────────────────
    st.info("👆 Upload a CSV or Excel file above to get started.")
    st.markdown("""
    ### How it works:
    | Step | Action |
    |------|--------|
    | 1️⃣ | Upload your CSV or Excel dataset |
    | 2️⃣ | Map the Phone, Email, and Address columns |
    | 3️⃣ | Choose a matching strategy from the sidebar |
    | 4️⃣ | Click **Run Clustering** and download the result |

    ### Matching Strategies:
    - **Match ANY (Broad):** Groups records if **any one** field is an exact match (Phone OR Email OR Address).
    - **Match ALL (Strict):** Groups records **only** if all three fields match exactly.
    """)