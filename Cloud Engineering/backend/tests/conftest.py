import pytest

def pytest_configure(config):
    # Set asyncio default fixture loop scope to 'function'
    config.option.asyncio_default_fixture_loop_scope = "function"
    
    # Only filter standard DeprecationWarning
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")
