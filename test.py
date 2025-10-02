import uuid
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# 1️⃣ Prepare the PDF generation code
# ----------------------------
pdf_code = """
from fpdf import FPDF

# Create a PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, "Hello from the sandbox!", ln=True)
pdf.cell(0, 10, "This PDF will be saved in the sandbox filesystem.", ln=True)

# Save PDF to sandbox filesystem
pdf_file_name = "sandbox_test.pdf"
pdf.output(pdf_file_name)

print("PDF created:", pdf_file_name)
"""

# ----------------------------
# 2️⃣ Create a sandbox and run code
# ----------------------------
session_id = str(uuid.uuid4())[:8]
print(f"Executing code in sandbox (Session: {session_id})...")

with Sandbox.create() as sandbox:
    # Run the PDF creation code
    execution = sandbox.run_code(pdf_code)
    
    # ✅ Check logs/output
    print("STDOUT:", execution.logs.stdout)
    print("TEXT:", execution.text)

    # ----------------------------
    # 3️⃣ Download the PDF from sandbox
    # ----------------------------
    local_file = f"local_sandbox_test.pdf"
    n =sandbox.files.read("sandbox_test.pdf") #("sandbox_test.pdf", local_file)
    print(f"PDF downloaded locally as: {n}")
