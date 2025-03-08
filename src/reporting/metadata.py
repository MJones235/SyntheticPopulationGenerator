from bs4 import BeautifulSoup

from src.llm_interface.base_llm import BaseLLM

def add_metadata_to_report(report_filename: str, metadata: dict, prompt: str, model: BaseLLM):
    """Inserts experiment metadata into an existing HTML report."""
    with open(report_filename, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Convert metadata to HTML
    metadata_html = "<div style='padding:12px;'><h2>Experiment Metadata</h2><ul>"
    for key, value in metadata.items():
        metadata_html += f"<li><strong>{key}:</strong> {value}</li>"
    metadata_html += "</ul>"

    metadata_html += f"""
    <h2>Model Configuration</h2>
    <pre style="background-color:#f4f4f4; padding:10px; border:1px solid #ccc;">{model.get_model_metadata()}</pre>

    <h2>Prompt Used</h2>
    <pre style="background-color:#f4f4f4; padding:10px; border:1px solid #ccc;">{prompt}</pre>
    </div>
    """

    # Insert metadata into the report
    soup.body.insert(0, BeautifulSoup(metadata_html, "html.parser"))

    # Save updated report
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(str(soup))
