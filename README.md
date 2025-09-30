# Project Setup & Usage

## Main Files
- **`modified_main.py`** → Entry point containing the CLI code for the bot.
- **`ui.py`** → UI interface for the CLI version.
- **`client/`** and **`server/`** → Contain partial MCP code for solving the problem.

## Environment Setup
1. Create a virtual environment with [uv](https://docs.astral.sh/uv/):

   ```bash
   
   uv venv
   ```


2. Activate the environment:


   ```bash
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```
3. Install dependencies from `requirements.txt`:

   ```bash
   uv pip install -r requirements.txt
   ```
4. Configure environment variables:

   * Create a `.env` file in the project root.
   * Add your API keys:

     * **Gemini Models API Key** (from Google AI Studio)
     * **E2B API Key**

## Directory Structure

When you run `modified_main.py`, two directories will be created automatically:

* `input_files/` → Place all original/input files here.
* `output_files/` → All generated files will be saved here.

Project tree (simplified):

```
├── client/
├── server/
├── input_files/
├── output_files/
├── modified_main.py
├── ui.py
├── fileprocessor.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── .env
```

## Running the Bot

Run the CLI version:

```bash
uv run modified_main.py
```

Run the UI version:

```bash
streamlit run ui.py
```
UI output 
<img width="1919" height="939" alt="image" src="https://github.com/user-attachments/assets/eeb48d99-2712-4845-a66a-fc42b74082e0" />


