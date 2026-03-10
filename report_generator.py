from fpdf import FPDF
from pathlib import Path
from datetime import datetime
from config import SpacecraftConfig

class PDFReport(FPDF):
    """Custom PDF class with Header and Footer."""
    
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33) # Placeholder for logo
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'SMART-DEORBIT System - Technical Report', border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='R')

class ReportGenerator:
    """Class to generate PDF reports for de-orbit missions."""
    
    def __init__(self, config: SpacecraftConfig, output_dir: Path = Path("outputs/reports")):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"Report_{config.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.pdf = None

    def create_base_document(self):
        """Initializes the PDF document with a new page."""
        self.pdf = PDFReport()
        self.pdf.alias_nb_pages()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def save_document(self):
        """Finalizes and saves the PDF document."""
        if self.pdf:
            self.pdf.output(str(self.output_path))
            return self.output_path
        return None
