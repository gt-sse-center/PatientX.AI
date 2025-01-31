# Installing Ollama Without Root Privileges

This document explains the changes made to the default **Ollama installation script** to allow installation **without root privileges** by setting up Ollama in a user-defined directory.

---

## **1. Why These Changes Were Made**
By default, the **Ollama installation script** installs the software in system-wide directories such as `/usr/local/bin`, requiring **root (sudo) access**. The modified script:
- Installs Ollama **entirely within a user-accessible directory**.
- Bypasses the need for **sudo** or system-wide changes.
- Allows users to **run Ollama locally** without administrative permissions.

---

## **2. Key Changes to the Installation Script**
### **A. Changing the Installation Directory**
Instead of installing Ollama in **`/usr/local/bin`**, we modified the script to install it in:

```bash
OLLAMA_INSTALL_DIR=/PATH/ollama
BINDIR=/PATH/ollama/bin
```

This ensures the installation remains in a user-writable location.
The bin/ directory is used for executables, making it easy to manage.

### B. Preventing the Script from Requiring Root
####	1.	Disabled Root Checks
The script originally exits if it detects that the user is not root. These checks have been removed.
####	2.	Avoiding sudo and install Commands
	•	Commands like install -o0 -g0 -m755 -d $BINDIR require root access.
	•	Instead, we replaced them with:

```bash
mkdir -p $BINDIR
```

	•	This allows directories to be created without requiring superuser privileges.

### C. Running Ollama Without System Services
```bash
$BINDIR/ollama serve
```
If needed, you can add it to your shell profile for easier access:
```bash
echo 'export PATH="/nethome/rfievet3/USERSCRATCH/projects/ollama/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```