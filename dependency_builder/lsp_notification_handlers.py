import logging
from typing import Dict, Any, Callable

def _clean_uri(uri: str) -> str:
    """Helper to convert LSP URI to file system path, handling encoding."""
    if not uri:
        return ""
    return uri.replace("file://", "").replace("%2B", "+")

def semantic_highlight_handler(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """
    Handles $ccls/publishSemanticHighlight notifications.
    Used to understand how the server is parsing symbols in real-time.
    """
    def handler(params: Dict[str, Any]):
        uri = params.get("uri", "")
        symbols = params.get("symbols", [])
        file_path = _clean_uri(uri)
        
        # Use debug to prevent log spam during massive indexing
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[SemanticHighlight] File: {file_path} | Symbols found: {len(symbols)}")

        if not symbols:
            return

        for sym in symbols:
            kind = sym.get("kind")
            role = sym.get("role")
            for r in sym.get("ranges", []):
                start = r.get("start", {})
                end = r.get("end", {})
                start_line = start.get('line')
                end_line = end.get('line')
                
                # Only log detailed symbol info at DEBUG level
                logger.debug(
                    f"  - Symbol: Kind={kind}, Role={role} | "
                    f"Range: L{start_line}:{start.get('character')} - L{end_line}:{end.get('character')}"
                )
    return handler


def skipped_ranges_handler(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """
    Handles $ccls/publishSkippedRanges notifications.
    Useful for identifying #if 0 blocks or inactive preprocessor branches.
    """
    def handler(params: Dict[str, Any]):
        uri = params.get("uri", "")
        skipped_ranges = params.get("skippedRanges", [])
        file_path = _clean_uri(uri)
        
        if not skipped_ranges:
            return

        logger.info(f"[SkippedRanges] File: {file_path} | Skipped Blocks: {len(skipped_ranges)}")

        for r in skipped_ranges:
            start = r.get("start", {})
            end = r.get("end", {})
            logger.info(f"  - Block skipped: Lines {start.get('line')} to {end.get('line')}")
    return handler


def progress_handler(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """
    Handles $/progress notifications.
    Tracks background indexing or compilation tasks.
    """
    def handler(params: Dict[str, Any]):
        token = params.get("token")
        value = params.get("value", {})
        kind = value.get("kind")
        title = value.get("title", "")
        message = value.get("message", "")
        percentage = value.get("percentage")

        if kind == "begin":
            logger.info(f"[LSP Progress] [{token}] Started: {title} {message}")
        elif kind == "report":
            pct_display = f" ({percentage}%)" if percentage is not None else ""
            # Log reports as debug unless they contain a significant message
            if message or percentage == 100:
                logger.info(f"[LSP Progress] [{token}] {message}{pct_display}")
            else:
                logger.debug(f"[LSP Progress] [{token}] Processing...{pct_display}")
        elif kind == "end":
            logger.info(f"[LSP Progress] [{token}] Completed: {message}")
    return handler


def work_done_progress_create_handler(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """
    Handles window/workDoneProgress/create requests.
    """
    def handler(params: Dict[str, Any]):
        token = params.get("token")
        logger.debug(f"[WorkDoneProgress] Token created: {token}")
    return handler