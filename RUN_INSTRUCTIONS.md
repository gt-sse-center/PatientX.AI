# Running the PatientX.AI Pipeline

This guide outlines the steps to set up and run **PatientX** on a server, including configuring a new installation directory, creating a virtual environment, and running the required scripts.

---

## 1. Creating a tmux Session (OPTIONAL but HIGHLY recommended)

Please reference [`tmux` explanation documentation](WHY_TMUX.md) for more details on why to use `tmux`

To create a new session, run

```bash
tmux new -s session_name
```

---

## 2. SSH into HPC Cluster (optional but recommended for large datasets, LLM interpretation)

SSH into the HPC platform where you would like to run the PatientX code

---

## 3. Creating a Virtual Environment in Python

Virtual environments are essential for managing dependencies and avoiding conflicts between different projects. This guide will walk you through creating and activating a virtual environment using `venv`.

---

### Prerequisites

- Make sure Python 3.10+ is installed on your system. You can check by running:

    ```bash
    python --version
    ```

    or

    ```bash
    python3 --version
    ```

---

### Creating a Virtual Environment

Virtual environments can help us manage dependecies across projects and ensure there are no conflicting dependencies. To create a virtual environment, use the following command:

```bash
python -m venv {ENV_NAME}
```

This will create a new directory called myenv containing the virtual environment.

---

### Activating the Virtual Environment

To activate the new virtual environment

**On Windows**
```powershell
{ENV_NAME}\Scripts\Activate
```

**On macOS and Linux**
```zsh
source {ENV_NAME}/bin/activate
```


## 4. Installing Dependencies

All dependecies and necessary packages for `PatientX.AI` can be installed by running

```
pip install -e .
```

---

## 5. Start Ollama Server (optional - for LLM representation model)

### Install Ollama

Install [ollama](https://ollama.com/) by either going through their documentation or, if you do not have sudo access, by running
```
./script/install.sh
```

### Run Ollama Server

Run ollama server in the background by running
```bash
ollama serve &
```

### Run LLM model
Run the chosen model using
```bash
ollama run model-name &
```

---

## 6. Run PatientX.AI Code

To run the PatientX.AI code

1. Update the config file with your desired parameters
2. Run the pipeline using
```bash
python src/PatientX/run.py <datapath> <resultpath>
```
3. For help, run
```bash
python src/PatientX/run.py --help
```

For further assistance, consult the **PatientX** documentation or contact the development team.
