from loaders.data_loader import load_data, preprocess_data


def test_preprocess_sample():
    grants = load_data(file_path='data/sample.json')
    df, grouped_df = preprocess_data(grants)
    assert not df.empty
    assert not grouped_df.empty
