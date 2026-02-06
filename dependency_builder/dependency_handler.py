import logging
import json
import os
from typing import Optional, Dict, Any, Union

# Internal imports
from dependency_builder.ccls_dependency_builder import CCLSDependencyBuilder

class DependencyFetcher:
    """
    Handler class to process dependency fetch requests.
    Dispatches requests to the CCLSDependencyBuilder based on endpoint type.
    Includes caching mechanism to avoid re-fetching existing data.
    """
    
    def __init__(self):
        pass

    def _get_artifact_path(self, output_dir: str, raw_filename: str) -> str:
        """
        Generates the full safe path for an artifact.
        """
        target_dir = output_dir if output_dir else "out"
        
        # Sanitize filename
        safe_filename = raw_filename.replace("/", "_").replace("\\", "_")
        
        # Remove very long filenames to prevent OS errors
        if len(safe_filename) > 200:
            safe_filename = safe_filename[:200] + ".json"
            
        return os.path.join(target_dir, safe_filename)

    def _load_cached_data(self, artifact_path: str, logger: logging.Logger) -> Optional[Any]:
        """
        Attempts to load data from an existing artifact file.
        Returns None if file does not exist.
        """
        if os.path.exists(artifact_path):
            try:
                logger.info(f"Cache Hit: Loading dependencies from {artifact_path}")
                with open(artifact_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache file {artifact_path}: {e}")
        return None

    def _save_data(self, artifact_path: str, data: Any, logger: logging.Logger) -> None:
        """
        Saves data to the artifact path, ensuring parent directories exist.
        """
        try:
            target_dir = os.path.dirname(artifact_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.debug(f"Saved artifact to {artifact_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save debug artifact {artifact_path}: {e}")
    
    def process_message(
        self,
        repo_path: str,
        data_store_path: str,
        unique_project_prefix: str,
        endpoint_type: str,
        function_name: Optional[str] = None,
        file_name: Optional[str] = None,
        start: Optional[Union[str, int]] = None,
        end: Optional[Union[str, int]] = None,
        line: Optional[Union[str, int]] = None,
        character: Optional[Union[str, int]] = None,
        level: Optional[Union[str, int]] = None,
        project_logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        
        # Ensure we have a logger
        if not project_logger:
            project_logger = logging.getLogger("DependencyFetcher")
            project_logger.setLevel(logging.INFO)

        # 1. Input Normalization
        try:
            level = int(level) if level is not None else 1
        except ValueError:
            level = 1

        # Resolve Absolute Paths
        project_root = os.path.abspath(repo_path)
        
        # Determine Cache Path
        if data_store_path:
            abs_output_dir = os.path.abspath(data_store_path)
            ccls_index_path = os.path.join(abs_output_dir, ".ccls-cache")
        else:
            # Fallback if no output dir provided
            ccls_index_path = os.path.join(project_root, "out", ".ccls-cache")

        # Input Sanitization
        if file_name and os.path.isabs(file_name):
            try:
                if file_name.startswith(project_root):
                    file_name = os.path.relpath(file_name, project_root)
            except ValueError:
                project_logger.warning(f"Could not relativize path: {file_name}")

        # 2. Health Check
        if "health_check" in endpoint_type:
            return {"message": "RUNNING OK", "data": {}}
        
        # Initialize the Builder (Lightweight init)
        fetcher = CCLSDependencyBuilder(project_root, ccls_index_path, project_logger)

        # 3. Handle Endpoints
        try:
            # --- Fetch by Component Name ---
            if "fetch_dependencies_by_component" in endpoint_type:
                if not function_name:
                    return {"message": "Error: Function name is mandatory", "data": {}}
                if not file_name:
                    return {"message": "Error: file_name is mandatory", "data": {}}
                
                # Check Cache
                artifact_name = f"fetch_comp_{os.path.basename(file_name)}_{function_name}.json"
                artifact_path = self._get_artifact_path(data_store_path, artifact_name)
                
                cached_data = self._load_cached_data(artifact_path, project_logger)
                if cached_data is not None:
                    status_msg = "success (cached)" if cached_data else "No dependencies found (cached)"
                    return {"message": status_msg, "data": cached_data}

                # Cache Miss - Run Fetcher
                fn_dependencies = fetcher.get_dependency_component(
                    file_path=file_name,
                    component_name=function_name,
                    level=level
                )
                
                data = fn_dependencies if fn_dependencies else {}
                status_msg = "success" if fn_dependencies else "No dependencies found"
                
                self._save_data(artifact_path, data, project_logger)
                return {"message": status_msg, "data": data}

            # --- Fetch by Line and Character (Position) ---
            elif "fetch_dependencies_by_line_character" in endpoint_type:
                
                if line is None:
                    return {"message": "Error: line is mandatory", "data": {}}
                if character is None:
                    return {"message": "Error: character is mandatory", "data": {}}
                if not file_name:
                    return {"message": "Error: file_name is mandatory", "data": {}}

                # Parse integers safely
                try:
                    line_no = int(line)
                    char_no = int(character)
                except ValueError:
                    return {"message": "Error: line and character must be integers", "data": {}}

                # Check Cache
                artifact_name = f"fetch_pos_{os.path.basename(file_name)}_{line_no}_{char_no}.json"
                artifact_path = self._get_artifact_path(data_store_path, artifact_name)

                cached_data = self._load_cached_data(artifact_path, project_logger)
                if cached_data is not None:
                    status_msg = "success (cached)" if cached_data else "No dependencies found (cached)"
                    return {"message": status_msg, "data": cached_data}

                # Cache Miss - Run Fetcher
                fn_dependencies = fetcher.get_dependency_line_char(
                    file_path=file_name, 
                    line_no=line_no,
                    character_no=char_no, 
                    level=level
                )
                
                data = fn_dependencies if fn_dependencies else {}
                status_msg = "success" if fn_dependencies else "No dependencies found"

                self._save_data(artifact_path, data, project_logger)
                return {"message": status_msg, "data": data}

            # --- Fetch by File (Range/Diff) ---
            elif "fetch_dependencies_by_file" in endpoint_type:
                if not file_name:
                    return {"message": "Error: file_name is mandatory", "data": {}}

                # Parse/Default start/end
                start_val = 0
                if start is not None:
                    try:
                        start_val = int(start)
                    except ValueError:
                        pass
                
                end_val = float('inf')
                if end is not None:
                    try:
                        end_val = int(end)
                    except ValueError:
                        pass

                # Check Cache
                artifact_name = f"fetch_file_{os.path.basename(file_name)}_{start_val}_{end_val}.json"
                artifact_path = self._get_artifact_path(data_store_path, artifact_name)

                cached_data = self._load_cached_data(artifact_path, project_logger)
                if cached_data is not None:
                    status_msg = "success (cached)" if cached_data else "No dependencies found (cached)"
                    return {"message": status_msg, "data": cached_data}

                # Cache Miss - Run Fetcher
                fn_dependencies = fetcher.get_dependency_diff(
                    file_path=file_name, 
                    start=start_val, 
                    end=end_val, 
                    level=level
                )
                
                data = fn_dependencies if fn_dependencies else []
                status_msg = "success" if fn_dependencies else "No dependencies found"
                
                self._save_data(artifact_path, data, project_logger)
                return {"message": status_msg, "data": data}

            else:
                return {"message": f"Error: Unknown endpoint type '{endpoint_type}'", "data": {}}

        except Exception as e:
            project_logger.exception(f"Critical error in dependency fetcher: {e}")
            return {"message": f"Error in dependencies fetch: {str(e)}", "data": {}}