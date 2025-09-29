
````markdown
# Project Setup & Usage

## Main File
The primary entry point is **`modified_main.py`**, which contains the CLI code for the bot.

## Environment Setup
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

3. Configure environment variables:

   * Create a `.env` file.
   * Add your API keys:

     * **Gemini Models API Key** (from Google AI Studio)
     * **E2B API Key**

## Directory Structure

When you run `modified_main.py`, two directories will be created automatically:

* `input_files/` → Place all original/input files here.
* `output_files/` → All generated files will be saved here.

## Running the Bot

```bash
python modified_main.py
```




