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

def test_report_optimization_results():
    """Test that optimization results are added to the report."""
    from delta_v_optimizer import OptimizationResult
    
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
    
    result = OptimizationResult(
        success=True,
        optimal_dv=45.2,
        lifetime_at_optimal=4.8,
        objective_value=0.05,
        n_function_evaluations=25,
        compute_time_s=12.5,
        method="hybrid",
        message="Converged successfully",
        dv_history=[0, 10, 20, 30, 40, 45.2],
        lifetime_history=[25.0, 15.0, 10.0, 7.0, 5.5, 4.8],
        no_burn_lifetime=25.0,
        propulsive_only_dv=150.0,
        fuel_savings_percent=69.8
    )
    
    generator = ReportGenerator(config)
    generator.create_base_document()
    generator.add_optimization_results(result)
    
    assert len(generator.pdf.pages) > 0

def test_report_plots(tmp_path):
    """Test that plots are added to the report."""
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
    
    # Create a dummy image for testing
    from PIL import Image
    import numpy as np
    img_path = tmp_path / "test_plot.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)
    
    generator = ReportGenerator(config)
    generator.create_base_document()
    generator.add_plots([img_path])
    
    assert len(generator.pdf.pages) > 0



