
# Steps to Run PatientX on a Server

This guide outlines the steps to set up and run **PatientX** on a server, including configuring a new installation directory, creating a Conda environment, and running the required scripts.

---

**Download the Miniconda Installer**  
   Use \`curl\` or \`wget\` to download the Miniconda installer:
   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```
   Or:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

## 1. Specify a New Installation Directory

To avoid issues with spaces in the directory path, configure a new installation directory for Miniconda:

1. **Abort the Current Installation**  
   If Miniconda installation is running, abort it by pressing `CTRL-C`.

2. **Restart the Installation**  
   Restart the installation and specify a directory without spaces. For example:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh -b -p /PATH/miniconda3_no_space -u
   ```

3. **Update the Shell’s Environment**  
   Add the new Miniconda directory to your PATH:
   ```bash
   echo 'export PATH="/YOUR_PATH/miniconda3_no_space/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

---

## 2. Create the Environment

Once Miniconda is installed, create a Conda environment for **PatientX**:

1. **Create a New Environment**  
   Create the environment with Python 3.9:
   ```bash
   conda create -n jupyter_env python=3.9
   ```

2. **Activate the Environment**  
   Activate the newly created environment:
   ```bash
   conda activate jupyter_env
   ```

---

## 3. Install Jupyter Notebook and Dependencies

1. **Install Jupyter Notebook**  
   Install Jupyter Notebook in the environment:
   ```bash
   conda install notebook
   ```


2. **Install Required Libraries**  
   Install `bertopic` and other dependencies:
   ```bash
   python -m pip install bertopic
   ```

---

## 4. Run and Convert the Notebook

1. **Export the Tokenizer Environment Variable**  
   To prevent warnings about tokenizer parallelism:
   ```bash
   export TOKENIZERS_PARALLELISM=false
   ```

2. **Convert the Notebook to a Python Script**  
   Run and convert the notebook `bertopic.ipynb`:
   ```bash
   jupyter nbconvert --to notebook --execute bertopic.ipynb
   ```

3. **Debugging Mode (Optional)**  
   If you need detailed logs while executing the notebook:
   ```bash
   jupyter nbconvert --to notebook --execute --debug bertopic.ipynb
   ```

---

## 5. Verify the Setup

After completing these steps:
- Ensure all dependencies are installed and the notebook executes successfully.
- Check for any errors in the logs during the `nbconvert` step.

For further assistance, consult the **PatientX** documentation or contact the development team.
