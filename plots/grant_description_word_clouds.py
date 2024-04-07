import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def grant_description_word_clouds(df, grouped_df, selected_chart, selected_role):

    if selected_chart == "Grant Description Word Clouds":
        st.header("Grant Description Word Clouds")
        st.write("Explore the most common words in grant descriptions based on different criteria.")

        cloud_basis_options = ['USD Cluster', 'Funder', 'Population', 'Strategy']
        selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

        column_mapping = {
            'USD Cluster': 'amount_usd_cluster',
            'Funder': 'funder_name',
            'Population': 'grant_population_tran',
            'Strategy': 'grant_strategy_tran'
        }
        selected_column = column_mapping[selected_basis]

        unique_values = grouped_df[selected_column].unique().tolist()

        selected_values = st.multiselect(f"Select {selected_basis}(s)", options=unique_values, default=unique_values)

        stopwords = set(STOPWORDS)
        additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the',
                                'The',
                                'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                                'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will',
                                'at',
                                'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it',
                                'It',
                                'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support',
                                'project',
                                'Project'}
        stopwords.update(additional_stopwords)

        charts_per_page = 10

        total_pages = (len(selected_values) + charts_per_page - 1) // charts_per_page

        page_number = st.selectbox("Select Page", options=list(range(1, total_pages + 1)))

        start_index = (page_number - 1) * charts_per_page
        end_index = min(start_index + charts_per_page, len(selected_values))

        for value in selected_values[start_index:end_index]:
            filtered_df = grouped_df[grouped_df[selected_column] == value]

            text = ' '.join(filtered_df['grant_description'])

            wordcloud = WordCloud(stopwords=stopwords, width=400, height=200).generate(text)

            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud for {selected_basis}: {value}')
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free up memory

            with col2:
                words = [word for word in text.split() if word.lower() not in stopwords]
                word_freq = pd.Series(words).value_counts()
                st.write(f"Top Words for {selected_basis}: {value}")
                st.write(word_freq.head(5))


        st.write("Type words into the box below to search all grant descriptions for specefic words.")
        input_text = st.text_input(
            "Enter word(s) to search (separate multiple words with commas):")

        view_mode = st.radio("Select View Mode:", ["Summary View", "Detailed View"])

        if input_text:
            search_terms = [term.strip() for term in input_text.split(',')]

            def contains_all_terms(description, terms):
                return all(term.lower() in description.lower() for term in terms)

            filtered_df = grouped_df[grouped_df[selected_column].isin(selected_values)]
            grant_descriptions = filtered_df[
                filtered_df['grant_description'].apply(contains_all_terms, args=(search_terms,))
            ]

            if not grant_descriptions.empty:
                st.write(f"Grant Descriptions containing all of the following terms: {', '.join(search_terms)}")
                for index, row in grant_descriptions.iterrows():
                    if view_mode == "Summary View":
                        st.markdown(f"- **{row['grant_description'][:50]}...**")
                    elif view_mode == "Detailed View":
                        st.markdown(f"**Grant Description:** {row['grant_description']}")
                        st.markdown(f"**Funder Name:** {row['funder_name']}")
                        st.markdown(f"**Funder City:** {row['funder_city']}")
                        # Make the funder profile URL clickable
                        if row['funder_profile_url']:
                            st.markdown(f"[**Funder Profile**]({row['funder_profile_url']})")
                        # Add more details as needed with good formatting
            else:
                st.write(
                    f"No grant descriptions found containing all of the following terms: {', '.join(search_terms)}.")