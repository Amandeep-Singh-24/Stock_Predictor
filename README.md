---
### README for Stock Analysis Project
==============================================================================================================================================================
#### Prerequisites
Before running the script, ensure you have Python 3 and an up-to-date version of pip installed. Follow these terminal-based steps according to your operating system:

##### Checking Python and Pip:

-**Common Step for All Systems**:
  - Open a terminal or command prompt.
  - Check Python version: `python3 --version` (or `python --version` on some Windows systems).
  - If Python 3.x is present, update pip: `python3 -m pip install --upgrade pip` (or `python -m pip install --upgrade pip` on some Windows systems).

IF THESE WORK YOU MAY CONTINUE TO RUN THE PROGRAM AND IT SHOULD BE FREE OF ERROR
==============================================================================================================================================================


##### MacOS Installation Guide (If nothing is downloaded in terms of python):

1. **Homebrew**:
   - Check for Homebrew: `brew --version`.
   - If not installed, paste this into the terminal:
     `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - Install Python 3: `brew install python3`.
   - Verify Python installation: `python3 --version`.

##### Windows Installation Guide (If nothing is downloaded in terms of python):

1. **Chocolatey**:
   - Open PowerShell as an administrator.
   - Check for Chocolatey: `choco -v`.
   - If not installed, paste this:
  
     `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))`

   - Install Python 3: `choco install python3`.
   - Verify Python installation: `python --version` or `python3 --version`.

#### Running the Script

1. **Download the Project**:
   - Ensure both `StockAnalysis.py` and `AAPL.csv` are in the same directory.

2. **Execute the Script**:
   - In the terminal, navigate to the project directory.
   - For MacOS/Linux: `python3 StockAnalysis.py`.
   - For Windows: `python StockAnalysis.py`.

##### Additional Tips:

-**Running on MacOS**:
  - Download the project folder to the desktop.
  - In Terminal, type `cd ` and drag the folder into Terminal, then press Enter.
  - Run: `python3 StockAnalysis.py`.

-**Using Visual Studio Code (VSCode)**:
  - Set the default interpreter path in VSCode to `python3`.
  - Open `StockAnalysis.py` and use the run button.

#### Output
The script will provide regression analysis results, including future AAPL stock price predictions, and recommend the best model based on RMSE, R², and future price considerations.

#### Note
This script serves educational purposes and should not be the sole basis for investment decisions. Stock market investments carry inherent risks, and historical data does not guarantee future performance.

---