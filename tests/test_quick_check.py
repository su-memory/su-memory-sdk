"""Quick test"""
import pytest
import sys
import os
import tempfile
import time

sys.path.insert(0, "src")

from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem


class TestQuick:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic(self):
        db_path = os.path.join(self.temp_dir, "test.db")
        backend = SQLiteBackend(db_path)
        
        memory = MemoryItem(
            id="test_1",
            content="Test",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        backend.add_memory(memory)
        
        retrieved = backend.get_memory("test_1")
        assert retrieved is not None
        assert retrieved.content == "Test"
        
        backend.close()
        print("Test passed!")
