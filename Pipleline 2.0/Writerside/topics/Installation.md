# Installation and Setup

This document describes the steps to set up and configure the Interactive Pipeline UI.

## Prerequisites

- **Python 3.11**: Ensure you have Python installed.We tested on `3.11.9`. Download it
  from [python.org](https://www.python.org/downloads/).
- **Pip**: Python package installer (usually included with Python).
- **Streamlit**: Used to run the interactive UI.
- **Other Dependencies**: Additional packages are required (see below).

## Environment Setup

1. **Download the Code**  

   Download the code from the __[One Drive Link](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/george_rahul_research_iiit_ac_in/EpfAWGEoSUdMsiaOd9N6Ok8BFLC1a_inkScN4xKkaKANuw?e=V1zl04)__.



2. **Create and Activate a Virtual Environment**

   It is recommended to use a virtual environment.

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

3. **Install the Required Packages**

   Install dependencies by running:

```Bash
pip install -r requirements.txt
```

4. **Configure the Environment Variables**

- First, create the `.env` file in case it doesn't exist:

```Bash
touch .env
```

- Now, open the `.env` file and add the following variables:

```Bash
GOOGLE_API_KEY=Your_Google_API_Key_Here
```