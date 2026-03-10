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

    def add_mission_data(self):
        """Adds mission configuration details to the report."""
        if not self.pdf:
            self.create_base_document()
            
        self.pdf.set_font('helvetica', 'B', 12)
        self.pdf.cell(0, 10, '1. Mission Configuration', new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font('helvetica', '', 10)
        
        data = [
            ('Mission Name', self.config.name),
            ('Mission Type', self.config.mission_type.value),
            ('Initial Altitude', f'{self.config.orbit_altitude/1000:.1f} km'),
            ('Inclination', f'{self.config.orbit_inclination:.2f} deg'),
            ('Dry Mass', f'{self.config.dry_mass:.1f} kg'),
            ('Wet Mass', f'{self.config.wet_mass:.1f} kg'),
            ('Drag Sail Area', f'{self.config.drag_sail_area:.1f} m²'),
            ('Drag Coefficient (Cd)', f'{self.config.Cd:.2f}'),
            ('Max Delta-V', f'{self.config.max_delta_v:.1f} m/s'),
            ('Specific Impulse', f'{self.config.specific_impulse:.1f} s'),
        ]
        
        for label, value in data:
            self.pdf.cell(50, 8, f'{label}:', border=0)
            self.pdf.cell(0, 8, f'{value}', border=0, new_x="LMARGIN", new_y="NEXT")
        
        self.pdf.ln(10)
