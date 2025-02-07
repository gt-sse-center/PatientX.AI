from unittest.mock import patch

from typer.testing import CliRunner
import pandas as pd
from pathlib import Path
import pytest

from PatientX.utils import read_csv_files_in_directory
from PatientX.run import app

dimensionality_reduction_models = [
    "pca",
    "umap",
    "randomname"
]

clustering_models = [
    "hdbscan",
    "kmeans",
    "agglomerative",
    "randomname"
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

result_df = pd.DataFrame({"result": ["result"]})

@pytest.mark.parametrize("dimensionality_reduction_model", dimensionality_reduction_models)
@pytest.mark.parametrize("clustering_model", clustering_models)
@pytest.mark.parametrize("save_embeddings", [False, True])
@pytest.mark.parametrize("document_diversity", document_diversity_values)
def test_run_to_completion(fs, dimensionality_reduction_model, clustering_model, save_embeddings, document_diversity):
    with patch("PatientX.run.get_representation_model", return_value=None) as mock_representation_model, \
        patch("PatientX.run.run_bertopic_model", return_value=(result_df, [1,2,3])) as mock_bertopic:
        repo_root = Path(__file__).parent.parent
        output_dir = Path("test_output")
        fs.create_dir(output_dir)

        input_dir = repo_root / "data" / "test_data"
        fs.add_real_directory(input_dir)

        output_file = output_dir / "output.csv"
        embeddings_file = output_dir / "embeddings.pkl"

        if save_embeddings:
            result = CliRunner().invoke(app, ["--datapath", input_dir, "--resultpath", output_dir, "--min-topic-size", 10, "--document-diversity", document_diversity, "--save-embeddings"])
        else:
            result = CliRunner().invoke(app,
                                        ["--datapath", input_dir, "--resultpath", output_dir, "--min-topic-size", 10,
                                         "--document-diversity", document_diversity, "--no-save-embeddings"])

        assert embeddings_file.exists() == save_embeddings
        assert result.exit_code == 0
        assert output_file.exists()

def test_read_csv_files_in_directory(fs):
    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "data" / "test_data"


    fs.create_dir("empty_dir")

    documents_empty_dir = read_csv_files_in_directory(Path("empty_dir"))

    assert len(documents_empty_dir) == 0

    fs.add_real_directory(input_dir)
    documents_csv = read_csv_files_in_directory(input_dir)

    # TODO: add comment explaining why this is 46
    assert len(documents_csv) == 46
