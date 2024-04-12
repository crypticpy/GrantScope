import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def grant_description_word_clouds(df, grouped_df, selected_chart, selected_role):
    if selected_chart == "Grant Description Word Clouds":
        st.header("Grant Description Word Clouds")
        st.write("""
        Welcome to the Grant Description Word Clouds page! This page provides insights into the most common words found in the grant descriptions.

        You can choose to view word clouds for the entire dataset or by specific clusters such as subject, population, or strategy. The system will automatically analyze the data and display the most significant word clouds.

        Additionally, you can search for specific words across all grant descriptions using the search box below. The search results will be displayed in a table format for easy exploration.
        """)

        stopwords = set(STOPWORDS)
        additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the',
                                'The',
                                'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                                'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will',
                                'at', 'Generally', 'generally', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which',
                                'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it',
                                'It', 'general', 'General', 'GENERAL', 'can', 'Can', 'more', 'More', 'has', 'Has', 'their',
                                'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support',
                                'project', 'grant', 'GRANT', 'Grant', 'funding', 'funded', 'funds', 'fund', 'funder', 'recipient', 'area',
                                'Project'}
        stopwords.update(additional_stopwords)

        cloud_basis_options = ['Entire Dataset', 'Subject', 'Population', 'Strategy', 'Funder', 'Recipient', 'Geographical Area', 'Description']
        selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

        if selected_basis == 'Entire Dataset':
            text = ' '.join(grouped_df['grant_description'])
            wordcloud = WordCloud(stopwords=stopwords, width=800, height=400).generate(text)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud for Entire Dataset')
            st.pyplot(fig)
            plt.close(fig)
        else:
            column_mapping = {
                'Subject': 'grant_subject_tran',
                'Population': 'grant_population_tran',
                'Strategy': 'grant_strategy_tran',
                'Funder': 'funder_name',
                'Recipient': 'recip_name',
                'Geographical Area': 'grant_geo_area_tran'
            }
            selected_column = column_mapping[selected_basis]

            top_values = grouped_df[selected_column].value_counts().nlargest(5).index.tolist()

            for value in top_values:
                filtered_df = grouped_df[grouped_df[selected_column] == value]
                text = ' '.join(filtered_df['grant_description'])
                wordcloud = WordCloud(stopwords=stopwords, width=400, height=200).generate(text)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud for {selected_basis}: {value}')
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Search Grant Descriptions")
        st.write("Type words into the box below to search all grant descriptions for specific words.")
        input_text = st.text_input("Enter word(s) to search (separate multiple words with commas):")

        if input_text:
            search_terms = [term.strip() for term in input_text.split(',')]

            def contains_all_terms(description, terms):
                return all(term.lower() in description.lower() for term in terms)

            grant_descriptions = grouped_df[
                grouped_df['grant_description'].apply(contains_all_terms, args=(search_terms,))]

            if not grant_descriptions.empty:
                st.write(f"Grant Descriptions containing all of the following terms: {', '.join(search_terms)}")
                grant_details = grant_descriptions[
                    ['grant_key', 'grant_description', 'funder_name', 'funder_city', 'funder_profile_url']]
                edited_data = st.data_editor(grant_details, num_rows="dynamic")

                if st.button("Download Search Results as Excel"):
                    output = download_excel(edited_data, "search_results.xlsx")
                    st.markdown(output, unsafe_allow_html=True)
            else:
                st.write(
                    f"No grant descriptions found containing all of the following terms: {', '.join(search_terms)}.")

        st.write("""
        We hope you find the word clouds and search functionality on the Grant Description Word Clouds page helpful in exploring the key themes and topics in the grant descriptions. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """)