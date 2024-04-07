import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

def grant_amount_heatmap(df, grouped_df, selected_chart, selected_role):

    if selected_chart == "Grant Amount Heatmap":
        st.header("Grant Amount Heatmap")
        st.write("Explore the intersection of grant dimensions and identify meaningful funding patterns.")

        dimension_options = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
        default_dim1, default_dim2 = dimension_options[:2]

        col1, col2 = st.columns(2)
        with col1:
            dimension1 = st.selectbox("Select Dimension 1", options=dimension_options,
                                      index=dimension_options.index(default_dim1))
        with col2:
            dimension2 = st.selectbox("Select Dimension 2", options=[d for d in dimension_options if d != dimension1],
                                      index=0)

        st.caption("Select individual values for each dimension to filter the heatmap.")

        col1, col2 = st.columns(2)
        with col1:
            with st.expander(f"Select {dimension1.split('_')[1].capitalize()}s", expanded=False):
                selected_values1 = st.multiselect(f"Select {dimension1.split('_')[1].capitalize()}s",
                                                  options=grouped_df[dimension1].unique(),
                                                  default=grouped_df[dimension1].unique())
        with col2:
            with st.expander(f"Select {dimension2.split('_')[1].capitalize()}s", expanded=False):
                selected_values2 = st.multiselect(f"Select {dimension2.split('_')[1].capitalize()}s",
                                                  options=grouped_df[dimension2].unique(),
                                                  default=grouped_df[dimension2].unique())

        pivot_table = grouped_df[
            grouped_df[dimension1].isin(selected_values1) &
            grouped_df[dimension2].isin(selected_values2)
            ].groupby([dimension1, dimension2])['amount_usd'].sum().unstack().fillna(0)

        fig = go.Figure(data=go.Heatmap(
            x=pivot_table.columns,
            y=pivot_table.index,
            z=pivot_table.values,
            colorscale='Plasma',
            hovertemplate='<b>%{yaxis.title.text}</b>: %{y}<br><b>%{xaxis.title.text}</b>: %{x}<br><b>Total Grant Amount</b>: %{z:,.0f}',
            colorbar=dict(title='Total Grant Amount')
        ))

        fig.update_layout(
            title=f'Total Grant Amount by {dimension1.split("_")[1].capitalize()} and {dimension2.split("_")[1].capitalize()}',
            xaxis_title=dimension2.split('_')[1].capitalize(),
            yaxis_title=dimension1.split('_')[1].capitalize(),
            width=800,
            height=800
        )

        st.plotly_chart(fig)

        if st.checkbox("Show Underlying Data"):
            st.write(pivot_table)

        if st.button("Download Data as Excel"):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            pivot_table.to_excel(writer, sheet_name='Heatmap Data')
            writer.close()
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="heatmap_data.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.subheader("Explore Grant Details")
        selected_value1 = st.selectbox(f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1)

        filtered_df = grouped_df[grouped_df[dimension1] == selected_value1]
        available_values2 = filtered_df[dimension2].unique().tolist()

        if available_values2:

            selected_value2 = st.selectbox(f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2)

            cell_grants = grouped_df[
                (grouped_df[dimension1] == selected_value1) &
                (grouped_df[dimension2] == selected_value2)
                ]

            if not cell_grants.empty:

                st.write(
                    f"Grants for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
                grant_details = cell_grants[['grant_key', 'grant_description', 'amount_usd']]
                st.write(grant_details)

                if st.button("Download Grant Details as CSV"):
                    csv = grant_details.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="grant_details.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:

                st.write(
                    f"No grants found for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
        else:

            st.write(
                f"No {dimension2.split('_')[1].capitalize()}s available for the selected {dimension1.split('_')[1].capitalize()}.")