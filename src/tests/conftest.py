"""
Test configuration and shared fixtures.
"""
import os
import sys
import pytest

# Add the src directory to Python path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
