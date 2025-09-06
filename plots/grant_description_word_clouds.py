import hashlib
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import Any, cast
try:  # optional dependency; available via transitive deps but handle gracefully
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    TFIDF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore
    TFIDF_AVAILABLE = False

try:
    from wordcloud import WordCloud, STOPWORDS  # type: ignore
    WORDCLOUD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    WordCloud = None  # type: ignore
    STOPWORDS = set()  # type: ignore
    WORDCLOUD_AVAILABLE = False

from utils.utils import download_excel, generate_page_prompt
from utils.chat_panel import chat_panel


def grant_description_word_clouds(df, grouped_df, selected_chart, selected_role, ai_enabled):
    # Guard clause
    if selected_chart != "Grant Description Word Clouds":
        return

    st.header("Grant Description Word Clouds")
    st.write(
        """
        Welcome to the Grant Description Word Clouds page! This page provides insights into the most common words found in the grant descriptions.

        You can choose to view word clouds for the entire dataset or by specific clusters such as subject, population, or strategy. The system will automatically analyze the data and display the most significant word clouds.

        Additionally, you can search for specific words across all grant descriptions using the search box below. The search results will be displayed in a table format for easy exploration.
        """
    )

    if not WORDCLOUD_AVAILABLE:
        st.info("WordCloud package is not installed. Please install 'wordcloud' to enable this page.")
        return

    # Ensure required columns exist and are clean
    if "grant_description" not in grouped_df.columns:
        st.error("Column 'grant_description' is missing from the dataset.")
        return
    grouped_df = grouped_df.copy()
    grouped_df["grant_description"] = grouped_df["grant_description"].fillna("")

    stopwords = set(STOPWORDS)
    # Case-insensitive deduped stopwords common in descriptions
    additional_stopwords = {
        'public', 'health', 'and', 'to', 'of', 'the', 'a', 'by', 'in', 'for', 'with', 'on', 'is', 'that', 'are',
        'as', 'be', 'this', 'will', 'generally', 'from', 'or', 'an', 'which', 'have', 'it', 'general', 'can',
        'more', 'has', 'their', 'not', 'who', 'we', 'support', 'project', 'grant', 'funding', 'funded', 'funds',
        'fund', 'funder', 'recipient', 'area'
    }
    stopwords.update({w.lower() for w in additional_stopwords})

    with st.expander("Word Cloud Settings", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            max_words = st.slider("Max words", min_value=50, max_value=1000, value=250, step=50)
            collocations = st.toggle("Show collocations (bigrams)", value=False)
            use_tfidf = st.toggle("Use TF-IDF weighting", value=False, disabled=not TFIDF_AVAILABLE,
                                   help=None if TFIDF_AVAILABLE else "scikit-learn not installed")
        with col_b:
            background_color = st.selectbox("Background", ["white", "black"], index=0)
            colormap = st.selectbox(
                "Colormap",
                [
                    "viridis", "plasma", "inferno", "magma", "cividis",
                    "Blues", "Greens", "Oranges", "Reds", "Purples",
                ],
                index=0,
            )
        with col_c:
            width = st.number_input("Width", min_value=300, max_value=2000, value=800, step=50)
            height = st.number_input("Height", min_value=200, max_value=1200, value=400, step=50)
        with col_d:
            min_word_len = st.slider("Min word length", min_value=2, max_value=8, value=3)
            extra_stop = st.text_input("Extra stopwords (comma-separated)")
        if extra_stop.strip():
            stopwords.update({s.strip().lower() for s in extra_stop.split(",") if s.strip()})

    # Regex to filter out short tokens
    word_regex = rf"[A-Za-z]{{{min_word_len},}}"

    @st.cache_data(show_spinner=False)
    def _cloud_png_bytes(text: str, width: int, height: int, max_words: int, background_color: str, colormap: str, collocations: bool, regex: str, stopwords_list: list[str]) -> bytes:
        # Build a stable key by hashing inputs (cache_data handles args, but content hashing makes it explicit)
        _ = hashlib.md5(
            (text + str(width) + str(height) + str(max_words) + background_color + colormap + str(collocations) + regex + ",".join(sorted(stopwords_list))).encode("utf-8")
        ).hexdigest()

        wc = WordCloud(
            stopwords=set(stopwords_list),
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            collocations=collocations,
            regexp=regex,
        ).generate(text)  # type: ignore

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @st.cache_data(show_spinner=False)
    def _cloud_png_bytes_from_freq(freq: dict[str, float], width: int, height: int, background_color: str, colormap: str) -> bytes:
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
        ).generate_from_frequencies(freq)  # type: ignore
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _tfidf_freqs(texts: list[str], stopwords_list: list[str], min_len: int) -> dict[str, float]:
        if not TFIDF_AVAILABLE or TfidfVectorizer is None:
            return {}
        token_pattern = rf"(?u)[A-Za-z]{{{min_len},}}"
        vec = TfidfVectorizer(stop_words=stopwords_list, token_pattern=token_pattern)
        try:
            X = vec.fit_transform(texts)
            # Sum TF-IDF scores across documents (cast to silence stubs on sparse types)
            sums = np.asarray(cast(Any, X).sum(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())
            freq = {t: float(s) for t, s in zip(terms, sums) if s > 0}
            # Normalize for display stability
            if freq:
                maxv = max(freq.values())
                if maxv > 0:
                    freq = {k: v / maxv for k, v in freq.items()}
            return freq
        except ValueError:
            # Empty vocabulary or all stopwords
            return {}

    cloud_basis_options = [
        "Entire Dataset",
        "Subject",
        "Population",
        "Strategy",
        "Funder",
        "Recipient",
        "Geographical Area",
        "Description",
    ]
    selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

    if selected_basis == "Entire Dataset":
        freqs: dict[str, float] = {}
        if use_tfidf:
            freqs = _tfidf_freqs(grouped_df["grant_description"].astype(str).tolist(), list(stopwords), min_word_len)
            if not freqs:
                st.info("No terms available for TF-IDF word cloud (possibly due to stopwords or short tokens). Falling back to raw text.")
                use_tfidf = False
        if use_tfidf:
            png = _cloud_png_bytes_from_freq(freqs, int(width), int(height), str(background_color), str(colormap))
        else:
            text = " ".join(grouped_df["grant_description"].astype(str))
            png = _cloud_png_bytes(
                text,
                int(width),
                int(height),
                int(max_words),
                str(background_color),
                str(colormap),
                bool(collocations),
                str(word_regex),
                list(stopwords),
            )
        st.image(png, caption="Word Cloud for Entire Dataset")
        st.download_button(
            "Download PNG",
            data=png,
            file_name="wordcloud_entire_dataset.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        column_mapping = {
            "Subject": "grant_subject_tran",
            "Population": "grant_population_tran",
            "Strategy": "grant_strategy_tran",
            "Funder": "funder_name",
            "Recipient": "recip_name",
            "Geographical Area": "grant_geo_area_tran",
        }
        selected_column = column_mapping[selected_basis]

        if selected_column not in grouped_df.columns:
            st.error(f"Column '{selected_column}' is missing from the dataset.")
            return

        top_n = st.slider("Top N categories", min_value=3, max_value=12, value=6)
        top_values = (
            grouped_df[selected_column]
            .fillna("")
            .astype(str)
            .value_counts()
            .nlargest(int(top_n))
            .index
            .tolist()
        )

        # Display in a responsive grid
        num_cols = 2 if width <= 600 else 3
        rows = int(np.ceil(len(top_values) / num_cols))
        idx = 0
        for _ in range(rows):
            cols = st.columns(num_cols)
            for c in cols:
                if idx >= len(top_values):
                    break
                value = top_values[idx]
                filtered_df = grouped_df[grouped_df[selected_column].astype(str) == str(value)]
                if use_tfidf:
                    freqs = _tfidf_freqs(filtered_df["grant_description"].astype(str).tolist(), list(stopwords), min_word_len)
                    if freqs:
                        png = _cloud_png_bytes_from_freq(freqs, max(int(width / num_cols), 300), int(height / 2), str(background_color), str(colormap))
                    else:
                        text = " ".join(filtered_df["grant_description"].astype(str))
                        png = _cloud_png_bytes(
                            text,
                            max(int(width / num_cols), 300),
                            int(height / 2),
                            int(max_words // 2),
                            str(background_color),
                            str(colormap),
                            bool(collocations),
                            str(word_regex),
                            list(stopwords),
                        )
                else:
                    text = " ".join(filtered_df["grant_description"].astype(str))
                    png = _cloud_png_bytes(
                        text,
                        max(int(width / num_cols), 300),
                        int(height / 2),
                        int(max_words // 2),
                        str(background_color),
                        str(colormap),
                        bool(collocations),
                        str(word_regex),
                        list(stopwords),
                    )
                with c:
                    st.image(png, caption=f"{selected_basis}: {value}")
                    st.download_button(
                        "Download PNG",
                        data=png,
                        file_name=f"wordcloud_{selected_basis.lower()}_{str(value).replace(' ', '_')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{selected_basis}_{idx}",
                    )
                idx += 1

    if ai_enabled:
        st.subheader("Grant Description — AI Assistant")
        st.write("Ask questions about the grant descriptions to gain insights and explore the data further.")
        additional_context = "the word clouds and search functionality for grant descriptions"
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)
        chat_panel(grouped_df, pre_prompt, state_key="wordclouds_chat", title="Word Clouds — AI Assistant")
    else:
        st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

    st.divider()

    st.subheader("Search Grant Descriptions")
    st.write("Type words into the box below to search all grant descriptions for specific words.")
    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_input("Enter word(s) (comma-separated)")
    with col2:
        match_type = st.selectbox("Match", options=["all", "any"], index=0)

    if input_text:
        terms = [t.strip().lower() for t in input_text.split(",") if t.strip()]
        series = grouped_df["grant_description"].astype(str).str.lower()
        if match_type == "all":
            mask = np.logical_and.reduce([series.str.contains(t, regex=False) for t in terms]) if terms else np.array([True] * len(series))
        else:
            mask = np.logical_or.reduce([series.str.contains(t, regex=False) for t in terms]) if terms else np.array([True] * len(series))
        results = grouped_df.loc[mask]

        if not results.empty:
            st.write(
                f"Grant descriptions containing {'all' if match_type=='all' else 'any'} of: {', '.join(terms)}"
            )
            cols_present = [c for c in ["grant_key", "grant_description", "funder_name", "funder_city", "funder_profile_url"] if c in results.columns]
            grant_details = results[cols_present]
            edited_data = st.data_editor(grant_details, num_rows="dynamic")
            if st.button("Download Search Results as Excel"):
                output = download_excel(edited_data, "search_results.xlsx")
                st.markdown(output, unsafe_allow_html=True)
        else:
            st.write("No grant descriptions matched your search.")

    st.write(
        """
        We hope you find the word clouds and search functionality on the Grant Description Word Clouds page helpful in exploring the key themes and topics in the grant descriptions. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """
    )

    st.markdown(
        """ This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                """
    )