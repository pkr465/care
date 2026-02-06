import os
import sys
import time
import shutil
import logging
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure the script can find your modules. 
# We assume this script is placed in the parent folder of 'dependency_builder'
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_ccls_path():
    """Ensures ccls is in the PATH."""
    if not shutil.which("ccls"):
        logger.error("❌ 'ccls' binary not found in PATH.")
        logger.error("👉 Run: export PATH=$PWD/Release:$PATH (or wherever your ccls binary is)")
        return False
    return True

def create_dummy_project(base_dir):
    """Creates a temporary C++ project to test indexing."""
    project_dir = base_dir / "test_ccls_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Header
    header_content = """
    #pragma once
    struct Vector2D {
        float x;
        float y;
    };
    """
    (project_dir / "math_utils.h").write_text(header_content)

    # 2. Create Source
    source_content = """
    #include "math_utils.h"
    
    int main() {
        Vector2D v;
        v.x = 10.0;
        return 0;
    }
    """
    (project_dir / "main.cpp").write_text(source_content)

    # 3. Create .ccls config (Crucial for correct indexing)
    # This tells ccls to treat files as C++ and look in current directory for headers
    ccls_config = "clang++\n-I."
    (project_dir / ".ccls").write_text(ccls_config)

    return project_dir

def run_test():
    logger.info("Starting Dependency Service Test...")

    # 1. Check Environment
    if not check_ccls_path():
        sys.exit(1)

    # 2. Verify Imports
    try:
        from dependency_builder.ccls_ingestion import CCLSIngestion
        from dependency_builder.dependency_service import DependencyService
        logger.info("✅ Imports successful.")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Ensure you are running this from the folder containing 'dependency_builder'.")
        sys.exit(1)

    # 3. Setup Data
    work_dir = Path(os.getcwd()) / "temp_test_env"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    project_path = "codebase"
    logger.info(f"✅ Created dummy project at: {project_path}")

    try:
        # 4. Test Ingestion
        logger.info("--- Testing CCLS Ingestion ---")
        ingestion = CCLSIngestion()
        
        # This should trigger 'ccls --index'
        # Adjust 'wait' if your class relies on background processing
        ingestion.initialize() 
        
        # Give ccls a moment to write cache if it's async (adjust based on your implementation)
        time.sleep(2) 
        
        cache_dir = project_path / ".ccls-cache"
        if cache_dir.exists():
            logger.info("✅ .ccls-cache generated successfully.")
        else:
            logger.warning("⚠️ .ccls-cache not found. Ingestion might have failed or is running quietly.")

        # 5. Test Dependency Service
        logger.info("--- Testing Dependency Fetch ---")
        # We want to find the definition of 'Vector2D' used in main.cpp
        target_file = str(project_path / "main.cpp")
        
        # Initialize Service
        service = DependencyService(repo_path=str(project_path))
        
        # Define a chunk of code (lines 4-6 where Vector2D is used)
        # Note: Line numbers are 0-based or 1-based depending on your implementation.
        # Assuming 1-based for the request.
        dependencies = service.get_dependencies(
            file_path=target_file,
            start_line=4,
            end_line=6
        )

        logger.info(f"Result: {dependencies}")

        # 6. Assertions
        # We expect to see 'struct Vector2D' in the output context
        if dependencies and "struct Vector2D" in str(dependencies):
            logger.info("✅ SUCCESS: Dependency Service resolved 'Vector2D' struct definition!")
        else:
            logger.error("❌ FAILURE: 'struct Vector2D' definition not found in response.")
            logger.error("Verify that ccls indexed the file correctly.")

    except Exception as e:
        logger.exception(f"❌ Exception occurred during test: {e}")
    
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)
        logger.info("Test Complete.")

if __name__ == "__main__":
    run_test()
