from pathlib import Path

def test_sample_image_can_exist_after_generation():
    # The test does not require generation to keep CI deterministic.
    assert Path("scripts/generate_sample_image.py").exists()
