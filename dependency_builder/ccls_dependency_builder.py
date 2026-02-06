import logging
import os
import xxhash
from collections import deque
from typing import Optional, Dict, Any, List, Union, Set
from urllib.parse import unquote, urlparse

# External imports
from pylspclient.lsp_pydantic_strcuts import TextDocumentItem, Position

# Internal imports
from dependency_builder.ccls_code_navigator import CCLSCodeNavigator

class CCLSDependencyBuilder:
    """
    Orchestrates the retrieval of code dependencies (functions, variables, types)
    using the CCLSCodeNavigator. Builds call graphs and token-level dependency trees.
    """
    
    def __init__(self, project_root: str, cache_path: str, logger: logging.Logger):
        self.project_root = os.path.abspath(project_root)
        self.cache_path = os.path.abspath(cache_path)
        self.logger = logger

    def _resolve_file_path(self, input_path: str) -> str:
        """
        Attempts to find the correct absolute path for a given input.
        Checks relative to project_root first.
        """
        if not input_path:
            return ""

        # 1. Try joining with project_root (Handle relative paths like './prplMesh/...')
        # We strip './' or leading slashes to ensure os.path.join works correctly
        cleaned_input = input_path.lstrip("." + os.sep)
        root_relative = os.path.abspath(os.path.join(self.project_root, cleaned_input))
        
        if os.path.exists(root_relative):
            return root_relative

        # 2. Try as a pure absolute path
        abs_input = os.path.abspath(input_path)
        if os.path.exists(abs_input):
            return abs_input

        # 3. Fallback: Return the root-relative version anyway
        return root_relative

    def _clean_uri(self, uri: str) -> str:
        """Converts LSP URI to Absolute file system path."""
        if not uri:
            return ""
        try:
            parsed = urlparse(uri)
            if parsed.scheme == "file":
                path = unquote(parsed.path) 
            else:
                path = unquote(uri)
            return os.path.abspath(path)
        except Exception:
            return uri

    def _get_dependency_bfs(self, nav: CCLSCodeNavigator, call_flow: Dict[str, Any], max_level: int = 1, is_parent: bool = False) -> Optional[Dict[str, Any]]:
        dependencies = {}
        q = deque()
        level = 0
        q.append(call_flow) 
        seen: Set[str] = set()
        
        while q:
            current_level_size = len(q)
            if level > max_level:
                break

            for _ in range(current_level_size):
                csym = q.popleft()
                if not csym or not csym.get("name") or csym["name"] in seen:
                    continue
                
                cdoc, cpos = nav.getDocandPosFromSymbol(csym)
                if not cdoc or not cpos:
                    continue
                
                if not nav.openDoc(cdoc):
                    continue
                
                try:
                    locs = nav.lsp_client.lsp_endpoint.call_method(
                        "textDocument/definition", textDocument=cdoc, position=cpos
                    )
                except Exception:
                    continue
                
                if not locs:
                    continue
                
                loc = locs[0]
                def_path = self._clean_uri(loc['uri'])
                ddoc = nav.create_doc(def_path)
                
                if not ddoc:
                    continue 
                
                dpos = Position(line=loc['range']['start']['line'], character=loc['range']['start']['character'])
                cdef = nav.getDefinition(ddoc, dpos)
                
                if level not in dependencies:
                    dependencies[level] = {}
                    
                cid = csym.get("id", xxhash.xxh64(f"{cdoc.uri}:{csym['name']}:{cpos.line}").hexdigest())
                
                dependencies[level][cid] = {
                    "name": csym["name"],
                    "definition": cdef,
                    "file": def_path,
                    "uri": ddoc.uri,
                    "start": loc['range']['start'],
                    "kind": "FUNCTION_DECL"
                }
                
                for child in csym.get("children", []):
                    q.append(child)
                seen.add(csym["name"])
            
            level += 1
            
        return dependencies

    def _fetch_dependencies_for_symbol(self, nav: CCLSCodeNavigator, doc: TextDocumentItem, pos: Position, level: int) -> Optional[Dict[str, Any]]:
        """Helper to fetch dependencies for a specific symbol position."""
        call_flow = nav.getCallee(doc, pos, level)
        definition = nav.getDefinition(doc, pos)
        component_name = nav.get_name(doc, pos)

        if not call_flow:
            call_flow = nav.create_call_flow_from_token(doc, pos)
        
        if not call_flow:
            call_flow = {
                "name": component_name,
                "location": {
                    "uri": doc.uri,
                    "range": {
                        "start": {"line": pos.line, "character": pos.character},
                        "end": {
                            "line": pos.line + (len(definition.splitlines()) if definition else 0),
                            "character": len(definition.splitlines()[-1]) if definition else 0
                        }
                    }
                }
            }
        
        successors = self._get_dependency_bfs(nav, call_flow, max_level=level)
        
        # Optional: Predecessors (Callers)
        parent_call_flow = nav.getCaller(doc, pos, level)
        predecessors = self._get_dependency_bfs(nav, parent_call_flow, max_level=level, is_parent=True) if parent_call_flow else {}

        return {
            "name": call_flow.get("name", component_name),
            "file": self._clean_uri(doc.uri),
            "definition": definition,
            "dependencies": {
                "successors": successors,
                "predecessors": predecessors
            }
        }

    def get_dependency_component(self, file_path: str, component_name: str, level: int = 1) -> Optional[Dict[str, Any]]:
        abs_file_path = self._resolve_file_path(file_path)
        nav = CCLSCodeNavigator(self.project_root, self.cache_path, self.logger)
        try:
            doc = nav.create_doc(abs_file_path)
            if not doc:
                self.logger.error(f"File not found: {abs_file_path}")
                return None
            
            nav.openDoc(doc)
            docSymbols = nav.getDocumentSymbolsKeySymbols(doc)
            
            if component_name not in docSymbols or not docSymbols[component_name]:
                self.logger.warning(f"Component '{component_name}' not found.")
                return None

            sym_data = docSymbols[component_name][0]
            doc, pos = nav.getDocandPosFromSymbol(sym_data)
            
            return self._fetch_dependencies_for_symbol(nav, doc, pos, level)
            
        except Exception as e:
            self.logger.exception(f"Error in get_dependency_component: {e}")
            return None
        finally:
            nav.killCCLSProcess()

    def get_dependency_line_char(self, file_path: str, line_no: int, character_no: int, level: int = 1) -> Optional[Dict[str, Any]]:
        abs_file_path = self._resolve_file_path(file_path)
        nav = CCLSCodeNavigator(self.project_root, self.cache_path, self.logger)
        try:
            doc = nav.create_doc(abs_file_path)
            if not doc:
                return None
            
            nav.openDoc(doc)
            pos = Position(line=line_no, character=character_no)
            
            return self._fetch_dependencies_for_symbol(nav, doc, pos, level)
        except Exception as e:
            self.logger.exception(f"Error in get_dependency_line_char: {e}")
            return None
        finally:
            nav.killCCLSProcess()

    def get_dependency_diff(self, file_path: str, start: int = 0, end: Union[int, float] = float("inf"), level: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch dependencies for ALL tokens found within a range of lines.
        Used for analyzing specific chunks of code (e.g., in git diffs or LLM context).
        """
        abs_file_path = self._resolve_file_path(file_path)
        nav = CCLSCodeNavigator(self.project_root, self.cache_path, self.logger)
        try:
            doc = nav.create_doc(abs_file_path)
            if not doc:
                self.logger.error(f"Could not open doc for diff: {abs_file_path}")
                return []
            
            nav.openDoc(doc)
            content = nav.read_file(abs_file_path)
            lines = content.splitlines()
            if not lines:
                return []

            end_idx = min(len(lines), int(end) + 1) if end != float("inf") else len(lines)
            diff_content = "\n".join(lines[start:end_idx])
            
            # Get tokens for the snippet
            tokens = nav.getTokens(diff_content, doc, Position(line=start, character=0))

            results = []
            seen_defs = set()
            
            for token in tokens.values():
                # Avoid duplicates
                if token["name"] in seen_defs:
                    continue
                
                # Resolve token definition
                try:
                    ddoc, dpos, _ = nav.getDefinitionFromToken(
                        doc, 
                        Position(line=token["line"], character=token.get("character", 0))
                    )
                except Exception:
                    continue
                
                if ddoc and dpos:
                    seen_defs.add(token["name"])
                    
                    # Fetch dependencies for this token's definition
                    deps = self._fetch_dependencies_for_symbol(nav, ddoc, dpos, level)
                    if deps:
                        results.append(deps)
                        
            return results
            
        except Exception as e:
            self.logger.exception(f"Error in get_dependency_diff: {e}")
            return []
        finally:
            nav.killCCLSProcess()