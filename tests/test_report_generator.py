import pytest
from pathlib import Path
from report_generator import ReportGenerator
from config import SpacecraftConfig, MissionType

def test_report_generator_initialization():
    """Test that ReportGenerator initializes correctly with a mission config."""
    config = SpacecraftConfig(
        name="Test Mission",
        mission_type=MissionType.PSLV_PS4,
        orbit_altitude=750e3,
        dry_mass=920.0,
        wet_mass=1000.0,
        cross_section_no_sail=2.0,
        drag_sail_area=10.0,
        Cd=2.2,
        max_delta_v=100.0,
        specific_impulse=250.0
    )
    
    generator = ReportGenerator(config)
    assert generator.config.name == "Test Mission"
    assert isinstance(generator.output_path, Path)

def test_report_header_footer():
    """Test that the PDF document is created with a header and footer."""
    config = SpacecraftConfig(
        name="Test Mission",
        mission_type=MissionType.PSLV_PS4,
        orbit_altitude=750e3,
        dry_mass=920.0,
        wet_mass=1000.0,
        cross_section_no_sail=2.0,
        drag_sail_area=10.0,
        Cd=2.2,
        max_delta_v=100.0,
        specific_impulse=250.0
    )
    
    generator = ReportGenerator(config)
    # This should trigger PDF creation logic
    generator.create_base_document()
    
    assert generator.pdf is not None
    assert generator.pdf.page_no() == 1

def test_report_save(tmp_path):
    """Test that the PDF document is saved to disk."""
    config = SpacecraftConfig(
        name="Test Mission",
        mission_type=MissionType.PSLV_PS4,
        orbit_altitude=750e3,
        dry_mass=920.0,
        wet_mass=1000.0,
        cross_section_no_sail=2.0,
        drag_sail_area=10.0,
        Cd=2.2,
        max_delta_v=100.0,
        specific_impulse=250.0
    )
    
    # Use tmp_path for testing
    generator = ReportGenerator(config, output_dir=tmp_path)
    generator.create_base_document()
    path = generator.save_document()
    
    assert path.exists()
    assert path.suffix == ".pdf"

def test_report_mission_data():
    """Test that mission data is added to the report."""
    config = SpacecraftConfig(
        name="Test Mission",
        mission_type=MissionType.PSLV_PS4,
        orbit_altitude=750e3,
        dry_mass=920.0,
        wet_mass=1000.0,
        cross_section_no_sail=2.0,
        drag_sail_area=10.0,
        Cd=2.2,
        max_delta_v=100.0,
        specific_impulse=250.0
    )
    
    generator = ReportGenerator(config)
    generator.create_base_document()
    generator.add_mission_data()
    
    # Check that some text was added (internal check of fpdf)
    assert len(generator.pdf.pages) > 0
    # We can't easily check PDF content without parsing, but we check it doesn't crash

