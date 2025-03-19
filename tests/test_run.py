from unittest.mock import patch

from typer.testing import CliRunner
import pandas as pd
from pathlib import Path
import pytest

from PatientX.utils import read_csv_files_in_directory, read_data_in_txt_file
from PatientX.run import app

dimensionality_reduction_models = [
    "pca",
    "umap",
]

clustering_models = [
    "hdbscan",
    "kmeans",
    "agglomerative",
]

save_embeddings_values = [
    False,
    True
]

document_diversity_values = [
    0,
    0.5,
    1.2
]

nr_rep_docs_values = [
    0,
    1,
    10,
    20
]

# List of tuples containing txt file contents and the expected return value of read_data_in_txt_file()
content_expected_pairs = [
    ("", []),
    ("a\nb\nc\nd\n", ["a\n", "b\n", "c\n", "d\n"]),
    ("\n", ["\n"]),
    ("ab\n\n", ["ab\n", "\n"]),
    ("!@\naf\n", ["!@\n", "af\n"])
]

result_df = pd.DataFrame({"result": ["result"]})


@pytest.mark.parametrize("nr_rep_docs_value", nr_rep_docs_values)
def test_num_rep_docs_args(fs, nr_rep_docs_value):
    with patch("PatientX.run.get_representation_model", return_value=None) as mock_representation_model, \
            patch("PatientX.run.run_bertopic_model",
                  return_value=(pd.DataFrame(), pd.DataFrame(), (pd.DataFrame(), pd.DataFrame()))) as mock_bertopic, \
            patch("PatientX.run.format_bertopic_results", return_value=pd.DataFrame()) as mock_bertopic_output:
        repo_root = Path(__file__).parent.parent
        output_dir = Path("test_output")
        fs.create_dir(output_dir)

        input_dir = repo_root / "data" / "test_data" / "pass"
        fs.add_real_directory(input_dir)

        result = CliRunner().invoke(app,
                                    [str(input_dir), str(output_dir), "--nr-representative-docs",
                                     nr_rep_docs_value])

        assert (result.exit_code != 0) == (nr_rep_docs_value == 0)


@pytest.mark.parametrize("dimensionality_reduction_model", dimensionality_reduction_models)
@pytest.mark.parametrize("clustering_model", clustering_models)
@pytest.mark.parametrize("save_embeddings", [False, True])
@pytest.mark.parametrize("document_diversity", document_diversity_values)
def test_run_to_completion(fs, dimensionality_reduction_model, clustering_model, save_embeddings, document_diversity):
    with patch("PatientX.run.get_representation_model", return_value=None) as mock_representation_model, \
            patch("PatientX.run.run_bertopic_model",
                  return_value=(pd.DataFrame(), pd.DataFrame(), (pd.DataFrame(), pd.DataFrame()))) as mock_bertopic, \
            patch("PatientX.run.format_bertopic_results", return_value=pd.DataFrame()) as mock_bertopic_output:
        repo_root = Path(__file__).parent.parent
        output_dir = Path("test_output")
        fs.create_dir(output_dir)

        input_dir = repo_root / "data" / "test_data" / "pass"
        fs.add_real_directory(input_dir)

        output_file = output_dir / "output.csv"
        embeddings_file = output_dir / "embeddings.pkl"

        bertopic_output_file = output_dir / "bertopic_final_results.csv"

        if save_embeddings:
            result = CliRunner().invoke(app,
                                        [str(input_dir), str(output_dir), "--min-topic-size", 10,
                                         "--document-diversity", document_diversity, "--save-embeddings"])
        else:
            result = CliRunner().invoke(app,
                                        [str(input_dir), str(output_dir), "--min-topic-size", 10,
                                         "--document-diversity", document_diversity, "--no-save-embeddings"])

        assert result.exit_code == 0
        assert embeddings_file.exists() == save_embeddings
        assert bertopic_output_file.exists()
        assert output_file.exists()


def test_read_csv_files_in_directory(fs):
    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "data" / "test_data" / "pass"

    fs.create_dir("empty_dir")

    docs = read_csv_files_in_directory(Path("empty_dir"))

    assert len(docs) == 0

    fs.add_real_directory(input_dir)
    docs = read_csv_files_in_directory(input_dir)

    # this should be 46 since loading datasets merges posts from the same forum, thread, and message number
    # the fake data has 46 unique datapoints
    assert len(docs) == 46


def test_read_csv_files_incorrect_structure():
    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "data" / "test_data" / "fail"
    missing_dir = repo_root / "data" / "test_data" / "missing_dir"

    # assert a csv files that does not adhere to the proper data format raises a KeyError
    with pytest.raises(KeyError, match="Check README file for proper data format"):
        read_csv_files_in_directory(input_dir)

    # assert something that is not a directory raises a NotADirectory error
    with pytest.raises(NotADirectoryError):
        read_csv_files_in_directory(missing_dir)


@pytest.mark.parametrize("content_expected_pair", content_expected_pairs)
def test_read_data_in_txt_file(fs, content_expected_pair):
    fs.create_file("file.txt", contents=content_expected_pair[0])

    read_content = read_data_in_txt_file(Path("file.txt"))

    assert read_content == content_expected_pair[1]


def test_read_data_in_txt_file_nonexistent_file(fs):
    assert read_data_in_txt_file(Path("file.txt")) == []
