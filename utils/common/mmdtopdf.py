import subprocess
import os
import img2pdf

class MermaidToPDFConverter:
    """
    Converts a Mermaid .mmd file to PNG using Mermaid CLI, then to PDF using img2pdf.
    Keeps both PNG and PDF files.
    """

    def __init__(self, mmdc_path="mmdc"):
        self.mmdc_path = mmdc_path  # Path to Mermaid CLI executable

    def mmd_to_png(self, mmd_file, png_file):
        """
        Uses Mermaid CLI to convert .mmd to .png.
        """
        out_dir = os.path.dirname(png_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        cmd = [
            self.mmdc_path,
            "-i", mmd_file,
            "-o", png_file,
            "--backgroundColor", "white"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"PNG generated: {png_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running Mermaid CLI: {e}")
            raise

    def png_to_pdf(self, png_file, pdf_file):
        """
        Converts PNG image to PDF using img2pdf.
        """
        out_dir = os.path.dirname(pdf_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        try:
            with open(pdf_file, "wb") as f:
                f.write(img2pdf.convert(png_file))
            print(f"PDF generated: {pdf_file}")
        except Exception as e:
            print(f"Error converting PNG to PDF: {e}")
            raise

    def convert(self, mmd_file, pdf_file, png_file="./out/temp_mermaid.png"):
        """
        Full conversion: .mmd -> .png -> .pdf
        Keeps both PNG and PDF files.
        Returns (png_file, pdf_file).
        """
        self.mmd_to_png(mmd_file, png_file)
        self.png_to_pdf(png_file, pdf_file)
        return png_file, pdf_file

# Example usage:
# converter = MermaidToPDFConverter()
# png_path, pdf_path = converter.convert("diagram.mmd", "diagram.pdf", "diagram.png")