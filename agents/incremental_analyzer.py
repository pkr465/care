# agents/incremental_analyzer.py
import os
import re
import time
import gc
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

# Import all analyzers
try:
    from agents.analyzers.dependency_analyzer import DependencyAnalyzer
    from agents.analyzers.complexity_analyzer import ComplexityAnalyzer
    from agents.analyzers.security_analyzer import SecurityAnalyzer
    from agents.analyzers.documentation_analyzer import DocumentationAnalyzer
    from agents.analyzers.quality_analyzer import QualityAnalyzer
    from agents.analyzers.maintainability_analyzer import MaintainabilityAnalyzer
    from agents.analyzers.test_coverage_analyzer import TestCoverageAnalyzer
    ANALYZERS_AVAILABLE = True
except ImportError as e:
    ANALYZERS_AVAILABLE = False
    console = Console()
    console.print(f"[yellow]Warning: Some analyzers not available: {e}[/yellow]")

# Initialize Rich console
console = Console()

# Global flag for graceful shutdown
shutdown_requested = False

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def force_garbage_collection():
    """Force garbage collection and log memory before/after."""
    memory_before = get_memory_usage()
    gc.collect()
    memory_after = get_memory_usage()
    freed = memory_before - memory_after
    if freed > 10:  # Only log if significant memory was freed
        console.print(f"[blue]🗑️  Garbage collection freed {freed:.1f}MB (was {memory_before:.1f}MB, now {memory_after:.1f}MB)[/blue]")
    return memory_after


class IncrementalCodebaseAnalyzer:
    """
    Enhanced incremental codebase analyzer that uses specialized analyzers
    and processes files in batches for memory efficiency.
    """
    
    def __init__(self, codebase_path: str, opts: dict):
        self.codebase_path = Path(codebase_path)
        self.opts = opts
        self.batch_size = opts.get('batch_size', 25)
        self.max_files = opts.get('max_files', 2000)
        self.memory_limit = opts.get('memory_limit', 3000)
        
        # Initialize analyzers
        self.analyzers = self._initialize_analyzers()
        
        # Track analyzer failures to reduce noise
        self.analyzer_failures = defaultdict(int)
        self.max_failure_warnings = 3  # Only show first 3 failures per analyzer type
        
        # Results storage
        self.file_analysis = {}
        self.dependency_graph = defaultdict(set)
        self.function_graph = defaultdict(dict)
        self.module_analysis = {}
        self.health_metrics = {}
        
        # Statistics
        self.processed_files = 0
        self.total_files = 0
        self.skipped_files = 0
        self.error_files = 0
        
        # File discovery
        self.cpp_files = []
        self.discover_files()
    
    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize all available analyzers."""
        analyzers = {}
        
        if not ANALYZERS_AVAILABLE:
            console.print("[yellow]⚠️  Analyzers not available, using basic analysis[/yellow]")
            return analyzers
        
        try:
            analyzers['dependency'] = DependencyAnalyzer(str(self.codebase_path))
            analyzers['complexity'] = ComplexityAnalyzer()
            analyzers['security'] = SecurityAnalyzer()
            analyzers['documentation'] = DocumentationAnalyzer()
            analyzers['quality'] = QualityAnalyzer()
            analyzers['maintainability'] = MaintainabilityAnalyzer()
            analyzers['test_coverage'] = TestCoverageAnalyzer()
            
            console.print(f"[green]✅ Initialized {len(analyzers)} specialized analyzers[/green]")
            console.print("[blue]ℹ️  Note: Some analyzers may return text results instead of structured data[/blue]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Error initializing analyzers: {e}[/yellow]")
            console.print("[yellow]Falling back to basic analysis[/yellow]")
        
        return analyzers
    
    def discover_files(self):
        """Discover all C/C++ files in the codebase."""
        console.print("[blue]🔍 Discovering C/C++ files...[/blue]")
        
        cpp_extensions = {'.c', '.cpp', '.cc', '.cxx', '.c++', '.h', '.hpp', '.hh', '.hxx', '.h++'}
        exclude_dirs = set(self.opts.get('exclude_dirs', []))
        exclude_globs = self.opts.get('exclude_globs', [])
        
        # Add common directories to exclude by default
        exclude_dirs.update({
            'build', 'builds', '.git', '.svn', '.hg', 'node_modules',
            'vendor', 'third_party', 'external', 'deps', 'dependencies',
            '__pycache__', '.pytest_cache', 'cmake-build-debug', 'cmake-build-release',
            'Debug', 'Release', 'x64', 'Win32', '.vs', 'bin', 'obj'
        })
        
        for ext in cpp_extensions:
            for file_path in self.codebase_path.rglob(f'*{ext}'):
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                
                # Skip if matches excluded glob
                if any(file_path.match(glob_pattern) for glob_pattern in exclude_globs):
                    continue
                
                # Skip very large files to prevent memory issues
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 5 * 1024 * 1024:  # 5MB limit
                        console.print(f"[yellow]⚠️  Skipping large file: {file_path.name} ({file_size/1024/1024:.1f}MB)[/yellow]")
                        self.skipped_files += 1
                        continue
                except OSError:
                    continue
                
                self.cpp_files.append(file_path)
                
                # Limit total files
                if len(self.cpp_files) >= self.max_files:
                    break
        
        self.total_files = len(self.cpp_files)
        console.print(f"[green]✅ Discovered {self.total_files} C/C++ files[/green]")
        
        if self.total_files == 0:
            raise ValueError("No C/C++ files found to analyze")
    
    def create_file_batches(self) -> List[List[Path]]:
        """Create batches of files for processing."""
        batches = []
        for i in range(0, len(self.cpp_files), self.batch_size):
            batch = self.cpp_files[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def analyze_file_batch(self, batch: List[Path], batch_num: int, total_batches: int) -> Dict[str, Any]:
        """Analyze a batch of files using specialized analyzers."""
        global shutdown_requested
        
        if shutdown_requested:
            return {"status": "cancelled"}
        
        console.print(f"[blue]🔧 Processing batch {batch_num}/{total_batches} ({len(batch)} files)[/blue]")
        
        batch_results = {
            'files': {},
            'dependencies': defaultdict(set),
            'functions': defaultdict(dict),
            'modules': {},
            'errors': []
        }
        
        start_memory = get_memory_usage()
        
        for file_path in batch:
            if shutdown_requested:
                break
                
            try:
                # Analyze individual file
                file_result = self.analyze_single_file(file_path)
                if file_result:
                    batch_results['files'][str(file_path)] = file_result
                    
                    # Extract dependencies for dependency graph
                    if 'dependencies' in file_result:
                        batch_results['dependencies'][str(file_path)] = set(file_result['dependencies'])
                    
                    # Extract functions for function graph
                    if 'functions' in file_result:
                        for func in file_result['functions']:
                            func_key = f"{file_path}::{func['name']}"
                            batch_results['functions'][func_key] = {
                                'file': str(file_path),
                                'name': func['name'],
                                'line': func.get('line', 0),
                                'complexity': func.get('cyclomatic_complexity', 1),
                                'dependencies': func.get('calls', [])
                            }
                    
                    self.processed_files += 1
                
                # Memory check after each file
                current_memory = get_memory_usage()
                if current_memory > self.memory_limit:
                    console.print(f"[yellow]⚠️  Memory limit reached ({current_memory:.1f}MB), forcing cleanup[/yellow]")
                    force_garbage_collection()
                    
            except Exception as e:
                error_msg = f"Error analyzing {file_path}: {str(e)}"
                batch_results['errors'].append(error_msg)
                console.print(f"[red]❌ {error_msg}[/red]")
                self.error_files += 1
        
        end_memory = get_memory_usage()
        memory_delta = end_memory - start_memory
        
        console.print(f"[green]✅ Batch {batch_num} complete: {len(batch_results['files'])} files processed[/green]")
        console.print(f"[blue]💾 Memory: {start_memory:.1f}MB → {end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)[/blue]")
        
        return batch_results
    
    def _safe_analyzer_call(self, analyzer, content: str, analyzer_name: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Safely call an analyzer and handle different return types."""
        try:
            # Check if analyzer has the analyze method
            if not hasattr(analyzer, 'analyze'):
                if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                    console.print(f"[yellow]⚠️  {analyzer_name} analyzer has no 'analyze' method[/yellow]")
                self.analyzer_failures[analyzer_name] += 1
                return None
            
            # Try calling the analyzer
            result = analyzer.analyze(content)
            
            # Handle different result types
            if result is None:
                return None
            elif isinstance(result, dict):
                # Ensure the dict has the expected structure
                normalized_result = {
                    'score': result.get('score', 50.0) if hasattr(result, 'get') else 50.0,
                    'issues': result.get('issues', []) if hasattr(result, 'get') else [],
                    'metrics': result.get('metrics', {}) if hasattr(result, 'get') else {},
                    'functions': result.get('functions', []) if hasattr(result, 'get') else [],
                    'classes': result.get('classes', []) if hasattr(result, 'get') else []
                }
                # Add any other fields from the original result
                if hasattr(result, 'items'):
                    for key, value in result.items():
                        if key not in normalized_result:
                            normalized_result[key] = value
                return normalized_result
            elif isinstance(result, str):
                # If it's a string, try to extract useful information
                string_result = result.strip()
                
                # Try to parse the string for useful information
                score = self._extract_score_from_string(string_result)
                issues = self._extract_issues_from_string(string_result, analyzer_name)
                
                # Only show warning for first few failures per analyzer type
                if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                    console.print(f"[yellow]⚠️  {analyzer_name} returned text result for {file_name} (using fallback processing)[/yellow]")
                self.analyzer_failures[analyzer_name] += 1
                
                return {
                    'raw_result': string_result,
                    'score': score,
                    'issues': issues,
                    'metrics': {},
                    'functions': [],
                    'classes': [],
                    'analyzer_type': 'text_based'
                }
            elif isinstance(result, (int, float)):
                # If it's a number, treat as score
                return {
                    'score': float(result),
                    'issues': [],
                    'metrics': {},
                    'functions': [],
                    'classes': [],
                    'analyzer_type': 'numeric'
                }
            elif isinstance(result, list):
                # If it's a list, assume it's issues or functions
                return {
                    'score': 50.0,
                    'issues': result if all(isinstance(item, dict) for item in result) else [],
                    'functions': result if all(isinstance(item, dict) and 'name' in item for item in result) else [],
                    'metrics': {},
                    'classes': [],
                    'analyzer_type': 'list_based'
                }
            else:
                # Unknown type, return None to trigger fallback
                if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                    console.print(f"[yellow]⚠️  {analyzer_name} returned unexpected type {type(result)} for {file_name}[/yellow]")
                self.analyzer_failures[analyzer_name] += 1
                return None
                
        except AttributeError as e:
            if "'str' object has no attribute 'get'" in str(e):
                if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                    console.print(f"[yellow]⚠️  {analyzer_name} returned string result for {file_name} (using fallback)[/yellow]")
            else:
                if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                    console.print(f"[yellow]⚠️  {analyzer_name} analysis failed for {file_name}: {e}[/yellow]")
            self.analyzer_failures[analyzer_name] += 1
            return None
        except Exception as e:
            if self.analyzer_failures[analyzer_name] < self.max_failure_warnings:
                console.print(f"[yellow]⚠️  {analyzer_name} analysis failed for {file_name}: {e}[/yellow]")
            self.analyzer_failures[analyzer_name] += 1
            return None
    
    def _extract_score_from_string(self, text: str) -> float:
        """Try to extract a numeric score from a text result."""
        # Look for patterns like "score: 75", "75%", "rating: 3.5/5", etc.
        score_patterns = [
            r'score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)%',
            r'rating[:\s]+(\d+(?:\.\d+)?)',
            r'quality[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/100',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*100'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-100 scale if needed
                    if score <= 1.0:
                        score *= 100
                    elif score <= 5.0:
                        score *= 20  # Convert 5-point scale to 100-point
                    return min(max(score, 0), 100)
                except ValueError:
                    continue
        
        # If no score found, return default based on text sentiment
        if any(word in text.lower() for word in ['good', 'excellent', 'high', 'well']):
            return 75.0
        elif any(word in text.lower() for word in ['poor', 'bad', 'low', 'terrible']):
            return 25.0
        else:
            return 50.0
    
    def _extract_issues_from_string(self, text: str, analyzer_name: str) -> List[Dict[str, Any]]:
        """Try to extract issues from a text result."""
        issues = []
        
        # Split text into lines and look for issue-like patterns
        lines = text.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Look for patterns that suggest issues
            if any(word in line.lower() for word in ['error', 'warning', 'issue', 'problem', 'violation']):
                severity = 'medium'
                if any(word in line.lower() for word in ['critical', 'severe', 'major']):
                    severity = 'high'
                elif any(word in line.lower() for word in ['minor', 'low', 'info']):
                    severity = 'low'
                
                issues.append({
                    'line': i,
                    'description': line,
                    'severity': severity,
                    'analyzer': analyzer_name,
                    'type': 'text_extracted'
                })
        
        return issues
    
    def analyze_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single file using specialized analyzers."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic file metadata
            result = {
                'file_relative_path': str(file_path.relative_to(self.codebase_path)),
                'file_name': file_path.name,
                'suffix': file_path.suffix,
                'size_bytes': len(content.encode('utf-8')),
                'language': self._detect_language(file_path),
                'source': content if self.opts.get('enable_vector_db', False) else None,
                'metrics': {
                    'total_lines': len(content.splitlines()),
                    'code_lines': self._count_code_lines(content),
                    'comment_lines': self._count_comment_lines(content),
                    'comment_ratio': self._calculate_comment_ratio(content),
                    'preprocessor_lines': self._count_preprocessor_lines(content)
                }
            }
            
            # Use specialized analyzers if available
            if self.analyzers:
                # Complexity analysis
                if 'complexity' in self.analyzers:
                    complexity_result = self._safe_analyzer_call(
                        self.analyzers['complexity'], content, 'Complexity', file_path.name
                    )
                    if complexity_result:
                        result['complexity_analysis'] = complexity_result
                        result['complexity_score'] = complexity_result.get('score', 0)
                        result['functions'] = complexity_result.get('functions', [])
                        result['classes'] = complexity_result.get('classes', [])
                    else:
                        result['complexity_score'] = self._calculate_complexity_basic(content)
                        result['functions'] = self._extract_functions_basic(content)
                        result['classes'] = self._extract_classes_basic(content)
                
                # Security analysis
                if 'security' in self.analyzers:
                    security_result = self._safe_analyzer_call(
                        self.analyzers['security'], content, 'Security', file_path.name
                    )
                    if security_result:
                        result['security_analysis'] = security_result
                        result['security_issues'] = security_result.get('issues', [])
                    else:
                        result['security_issues'] = self._detect_security_issues_basic(content)
                
                # Documentation analysis
                if 'documentation' in self.analyzers:
                    doc_result = self._safe_analyzer_call(
                        self.analyzers['documentation'], content, 'Documentation', file_path.name
                    )
                    if doc_result:
                        result['documentation_analysis'] = doc_result
                        result['documentation_score'] = doc_result.get('score', 0)
                    else:
                        result['documentation_score'] = self._calculate_documentation_score_basic(content)
                
                # Quality analysis
                if 'quality' in self.analyzers:
                    quality_result = self._safe_analyzer_call(
                        self.analyzers['quality'], content, 'Quality', file_path.name
                    )
                    if quality_result:
                        result['quality_analysis'] = quality_result
                        result['code_quality_metrics'] = quality_result.get('metrics', {})
                    else:
                        result['code_quality_metrics'] = self._calculate_quality_metrics_basic(content)
                
                # Maintainability analysis
                if 'maintainability' in self.analyzers:
                    maint_result = self._safe_analyzer_call(
                        self.analyzers['maintainability'], content, 'Maintainability', file_path.name
                    )
                    if maint_result:
                        result['maintainability_analysis'] = maint_result
                        result['maintainability_score'] = maint_result.get('score', 0)
                    else:
                        result['maintainability_score'] = 50.0  # Default score
                
                # Extract dependencies
                result['dependencies'] = self._extract_dependencies(content)
            else:
                # Fallback to basic analysis
                result.update({
                    'dependencies': self._extract_dependencies(content),
                    'functions': self._extract_functions_basic(content),
                    'classes': self._extract_classes_basic(content),
                    'complexity_score': self._calculate_complexity_basic(content),
                    'documentation_score': self._calculate_documentation_score_basic(content),
                    'security_issues': self._detect_security_issues_basic(content),
                    'code_quality_metrics': self._calculate_quality_metrics_basic(content)
                })
            
            return result
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing file {file_path}: {e}[/red]")
            return None
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect the programming language of a file."""
        suffix = file_path.suffix.lower()
        if suffix in {'.h', '.hpp', '.hh', '.hxx', '.h++'}:
            return 'cpp_header'
        elif suffix in {'.c'}:
            return 'c'
        elif suffix in {'.cpp', '.cc', '.cxx', '.c++'}:
            return 'cpp'
        return 'unknown'
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract #include dependencies from file content."""
        dependencies = []
        include_pattern = r'#include\s*[<"](.*?)[>"]'
        
        for match in re.finditer(include_pattern, content):
            include_file = match.group(1)
            dependencies.append(include_file)
        
        return dependencies
    
    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines."""
        lines = content.splitlines()
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                code_lines += 1
        
        return code_lines
    
    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines."""
        lines = content.splitlines()
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                comment_lines += 1
        
        return comment_lines
    
    def _calculate_comment_ratio(self, content: str) -> float:
        """Calculate comment to total lines ratio."""
        lines = content.splitlines()
        if not lines:
            return 0.0
        
        comment_lines = self._count_comment_lines(content)
        return comment_lines / len(lines)
    
    def _count_preprocessor_lines(self, content: str) -> int:
        """Count preprocessor directive lines."""
        lines = content.splitlines()
        preprocessor_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                preprocessor_lines += 1
        
        return preprocessor_lines
    
    # ... rest of the methods remain the same ...
    def _extract_functions_basic(self, content: str) -> List[Dict[str, Any]]:
        """Basic function extraction (fallback)."""
        functions = []
        patterns = [
            r'(?:^|\n)\s*(?:(?:static|inline|virtual|explicit|const|constexpr)\s+)*(?:\w+(?:\s*\*)*\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?(?:\s*->\s*\w+\s*)?\s*\{',
            r'(?:^|\n)\s*(?:(?:explicit|virtual)\s+)*(\w+)\s*\([^)]*\)\s*(?::\s*[^{]*)?(?:noexcept\s*)?\s*\{',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_name = match.group(1)
                if func_name and func_name not in {'if', 'for', 'while', 'switch', 'catch', 'try'}:
                    line_num = content[:match.start()].count('\n') + 1
                    functions.append({
                        'name': func_name,
                        'line': line_num,
                        'start_line': line_num,
                        'type': 'function',
                        'cyclomatic_complexity': 1  # Default complexity
                    })
        
        return functions
    
    def _extract_classes_basic(self, content: str) -> List[Dict[str, Any]]:
        """Basic class extraction (fallback)."""
        classes = []
        patterns = [
            r'(?:^|\n)\s*class\s+(\w+)(?:\s*:\s*[^{]*)?(?:\s*final\s*)?\s*\{',
            r'(?:^|\n)\s*struct\s+(\w+)(?:\s*:\s*[^{]*)?(?:\s*final\s*)?\s*\{',
            r'(?:^|\n)\s*union\s+(\w+)\s*\{',
            r'(?:^|\n)\s*enum\s+(?:class\s+)?(\w+)(?:\s*:\s*\w+)?\s*\{'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                class_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                classes.append({
                    'name': class_name,
                    'line': line_num,
                    'type': 'class'
                })
        
        return classes
    
    def _calculate_complexity_basic(self, content: str) -> float:
        """Basic complexity calculation (fallback)."""
        lines = content.splitlines()
        complexity_patterns = [
            r'\bif\s*\(',
            r'\belse\s+if\s*\(',
            r'\bfor\s*\(',
            r'\bwhile\s*\(',
            r'\bdo\s*\{',
            r'\bswitch\s*\(',
            r'\bcase\s+',
            r'\bcatch\s*\(',
            r'\btry\s*\{',
            r'\?\s*.*\s*:',
            r'&&',
            r'\|\|'
        ]
        
        complexity_count = 0
        for line in lines:
            line = line.strip()
            for pattern in complexity_patterns:
                complexity_count += len(re.findall(pattern, line))
        
        if len(lines) > 0:
            return min(complexity_count / len(lines) * 100, 100.0)
        return 0.0
    
    def _calculate_documentation_score_basic(self, content: str) -> float:
        """Basic documentation score calculation (fallback)."""
        lines = content.splitlines()
        comment_lines = 0
        doc_comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                comment_lines += 1
                if any(doc_marker in stripped for doc_marker in ['///', '/**', '@param', '@return', '@brief']):
                    doc_comment_lines += 1
        
        if len(lines) > 0:
            base_score = min(comment_lines / len(lines) * 100, 100.0)
            doc_bonus = min(doc_comment_lines / len(lines) * 50, 25.0)
            return min(base_score + doc_bonus, 100.0)
        return 0.0
    
    def _detect_security_issues_basic(self, content: str) -> List[Dict[str, Any]]:
        """Basic security issue detection (fallback)."""
        security_issues = []
        security_patterns = [
            (r'\bstrcpy\s*\(', 'Use of unsafe strcpy function', 'high'),
            (r'\bstrcat\s*\(', 'Use of unsafe strcat function', 'high'),
            (r'\bsprintf\s*\(', 'Use of unsafe sprintf function', 'high'),
            (r'\bgets\s*\(', 'Use of unsafe gets function', 'critical'),
            (r'\bsystem\s*\(', 'Use of system() function - command injection risk', 'high'),
        ]
        
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            for pattern, description, severity in security_patterns:
                if re.search(pattern, line):
                    security_issues.append({
                        'line': i,
                        'description': description,
                        'severity': severity,
                        'code': line.strip()
                    })
        
        return security_issues
    
    def _calculate_quality_metrics_basic(self, content: str) -> Dict[str, Any]:
        """Basic code quality metrics calculation (fallback)."""
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        long_lines = [line for line in lines if len(line) > 120]
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'long_lines_count': len(long_lines),
            'long_lines_percentage': len(long_lines) / len(lines) * 100 if lines else 0,
        }
    
    def merge_batch_results(self, batch_results: List[Dict[str, Any]]):
        """Merge results from all batches."""
        console.print("[blue]🔧 Merging batch results...[/blue]")
        
        for batch in batch_results:
            if batch.get('status') == 'cancelled':
                continue
                
            # Merge file analysis
            self.file_analysis.update(batch.get('files', {}))
            
            # Merge dependencies
            for file_path, deps in batch.get('dependencies', {}).items():
                self.dependency_graph[file_path].update(deps)
            
            # Merge function graph
            for func_key, func_data in batch.get('functions', {}).items():
                self.function_graph[func_key] = func_data
        
        console.print(f"[green]✅ Merged results from {len(batch_results)} batches[/green]")
        
        # Print analyzer failure summary
        if self.analyzer_failures:
            console.print(f"\n[blue]📊 Analyzer Performance Summary:[/blue]")
            for analyzer_name, failure_count in self.analyzer_failures.items():
                console.print(f"  {analyzer_name}: {failure_count} files used fallback processing")
    
    def _safe_get_score(self, data: Any, default: float = 0.0) -> float:
        """Safely extract a score from analyzer data."""
        if isinstance(data, dict):
            return data.get('score', default)
        elif isinstance(data, (int, float)):
            return float(data)
        else:
            return default
    
    def _safe_get_issues(self, data: Any) -> List[Dict[str, Any]]:
        """Safely extract issues from analyzer data."""
        if isinstance(data, dict):
            issues = data.get('issues', [])
            return issues if isinstance(issues, list) else []
        else:
            return []
    
    def calculate_health_metrics(self) -> Dict[str, Any]:
        """Calculate overall health metrics using specialized analyzers."""
        console.print("[blue]🔧 Calculating health metrics...[/blue]")
        
        if not self.file_analysis:
            return {'error': 'No files analyzed'}
        
        health_metrics = {}
        
        # Use dependency analyzer if available - check for correct method
        if 'dependency' in self.analyzers:
            dependency_analyzer = self.analyzers['dependency']
            
            # Try different method names that might exist
            dependency_method = None
            for method_name in ['analyze', 'analyze_dependencies', 'calculate_dependencies', 'run_analysis']:
                if hasattr(dependency_analyzer, method_name):
                    dependency_method = getattr(dependency_analyzer, method_name)
                    break
            
            if dependency_method:
                try:
                    # Convert dependency graph to format expected by analyzer
                    dep_graph = {}
                    for file_path, deps in self.dependency_graph.items():
                        dep_graph[file_path] = {
                            'dependencies': list(deps),
                            'external': False
                        }
                    
                    # Convert function graph to format expected by analyzer
                    func_graph = {}
                    for func_key, func_data in self.function_graph.items():
                        func_graph[func_key] = {
                            'dependencies': func_data.get('dependencies', []),
                            'file': func_data.get('file', ''),
                            'external': False
                        }
                    
                    # Try calling with different argument patterns
                    dependency_result = None
                    try:
                        dependency_result = dependency_method(dep_graph, func_graph)
                    except TypeError:
                        try:
                            dependency_result = dependency_method(dep_graph)
                        except TypeError:
                            try:
                                dependency_result = dependency_method()
                            except Exception as e:
                                console.print(f"[yellow]⚠️  All dependency analysis call patterns failed: {e}[/yellow]")
                    
                    if dependency_result and isinstance(dependency_result, dict):
                        health_metrics['dependency_score'] = dependency_result
                    else:
                        health_metrics['dependency_score'] = {'score': 50.0, 'grade': 'C'}
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Dependency analysis failed: {e}[/yellow]")
                    health_metrics['dependency_score'] = {'score': 50.0, 'grade': 'C'}
            else:
                console.print(f"[yellow]⚠️  No suitable method found on dependency analyzer[/yellow]")
                health_metrics['dependency_score'] = {'score': 50.0, 'grade': 'C'}
        
        # Aggregate other analyzer results
        if self.analyzers:
            # Safely aggregate complexity scores
            complexity_scores = []
            security_issues = []
            doc_scores = []
            quality_scores = []
            maintainability_scores = []
            
            for file_data in self.file_analysis.values():
                # Safely extract complexity scores
                if 'complexity_analysis' in file_data:
                    score = self._safe_get_score(file_data['complexity_analysis'])
                    if score > 0:
                        complexity_scores.append(score)
                elif 'complexity_score' in file_data:
                    score = self._safe_get_score(file_data['complexity_score'])
                    if score > 0:
                        complexity_scores.append(score)
                    
                # Safely extract security issues
                if 'security_analysis' in file_data:
                    issues = self._safe_get_issues(file_data['security_analysis'])
                    security_issues.extend(issues)
                elif 'security_issues' in file_data:
                    issues = file_data.get('security_issues', [])
                    if isinstance(issues, list):
                        security_issues.extend(issues)
                    
                # Safely extract documentation scores
                if 'documentation_analysis' in file_data:
                    score = self._safe_get_score(file_data['documentation_analysis'])
                    if score > 0:
                        doc_scores.append(score)
                elif 'documentation_score' in file_data:
                    score = self._safe_get_score(file_data['documentation_score'])
                    if score > 0:
                        doc_scores.append(score)
                    
                # Safely extract quality scores
                if 'quality_analysis' in file_data:
                    score = self._safe_get_score(file_data['quality_analysis'])
                    if score > 0:
                        quality_scores.append(score)
                    
                # Safely extract maintainability scores
                if 'maintainability_analysis' in file_data:
                    score = self._safe_get_score(file_data['maintainability_analysis'])
                    if score > 0:
                        maintainability_scores.append(score)
                elif 'maintainability_score' in file_data:
                    score = self._safe_get_score(file_data['maintainability_score'])
                    if score > 0:
                        maintainability_scores.append(score)
            
            # Calculate aggregate scores with safe defaults
            health_metrics.update({
                'complexity_score': {
                    'score': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 50.0,
                    'grade': self._score_to_grade(sum(complexity_scores) / len(complexity_scores) if complexity_scores else 50.0)
                },
                'documentation_score': {
                    'score': sum(doc_scores) / len(doc_scores) if doc_scores else 0.0,
                    'grade': self._score_to_grade(sum(doc_scores) / len(doc_scores) if doc_scores else 0.0)
                },
                'quality_score': {
                    'score': sum(quality_scores) / len(quality_scores) if quality_scores else 50.0,
                    'grade': self._score_to_grade(sum(quality_scores) / len(quality_scores) if quality_scores else 50.0)
                },
                'maintainability_score': {
                    'score': sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 50.0,
                    'grade': self._score_to_grade(sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 50.0)
                },
                'security_score': {
                    'score': max(0, 100 - len([i for i in security_issues if isinstance(i, dict) and i.get('severity') in ['critical', 'high']]) * 10),
                    'grade': self._score_to_grade(max(0, 100 - len([i for i in security_issues if isinstance(i, dict) and i.get('severity') in ['critical', 'high']]) * 10)),
                    'critical_issues': len([i for i in security_issues if isinstance(i, dict) and i.get('severity') == 'critical']),
                    'high_issues': len([i for i in security_issues if isinstance(i, dict) and i.get('severity') == 'high']),
                    'total_issues': len(security_issues)
                }
            })
        else:
            # Fallback to basic calculations
            health_metrics = self._calculate_basic_health_metrics()
        
        # Calculate overall health score
        scores = [
            health_metrics.get('dependency_score', {}).get('score', 0),
            health_metrics.get('complexity_score', {}).get('score', 0),
            health_metrics.get('documentation_score', {}).get('score', 0),
            health_metrics.get('quality_score', {}).get('score', 0),
            health_metrics.get('maintainability_score', {}).get('score', 0),
            health_metrics.get('security_score', {}).get('score', 0)
        ]
        
        valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > 0]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        health_metrics['overall_health'] = {
            'score': overall_score,
            'grade': self._score_to_grade(overall_score)
        }
        
        # Add statistics
        health_metrics['statistics'] = {
            'total_files': len(self.file_analysis),
            'total_lines': sum(file_data.get('metrics', {}).get('total_lines', 0) for file_data in self.file_analysis.values() if isinstance(file_data.get('metrics'), dict)),
            'total_functions': sum(len(file_data.get('functions', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('functions'), list)),
            'total_classes': sum(len(file_data.get('classes', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('classes'), list)),
            'processed_files': self.processed_files,
            'skipped_files': self.skipped_files,
            'error_files': self.error_files
        }
        
        return health_metrics
    
    def _calculate_basic_health_metrics(self) -> Dict[str, Any]:
        """Calculate basic health metrics when analyzers are not available."""
        # Aggregate basic metrics
        total_lines = sum(file_data.get('metrics', {}).get('total_lines', 0) for file_data in self.file_analysis.values() if isinstance(file_data.get('metrics'), dict))
        total_functions = sum(len(file_data.get('functions', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('functions'), list))
        total_classes = sum(len(file_data.get('classes', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('classes'), list))
        
        # Calculate average scores
        complexity_scores = [self._safe_get_score(file_data.get('complexity_score', 0)) for file_data in self.file_analysis.values()]
        doc_scores = [self._safe_get_score(file_data.get('documentation_score', 0)) for file_data in self.file_analysis.values()]
        
        complexity_scores = [s for s in complexity_scores if s > 0]
        doc_scores = [s for s in doc_scores if s > 0]
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        avg_documentation = sum(doc_scores) / len(doc_scores) if doc_scores else 0
        
        # Security analysis
        all_security_issues = []
        for file_data in self.file_analysis.values():
            issues = file_data.get('security_issues', [])
            if isinstance(issues, list):
                all_security_issues.extend(issues)
        
        critical_issues = len([issue for issue in all_security_issues if isinstance(issue, dict) and issue.get('severity') == 'critical'])
        high_issues = len([issue for issue in all_security_issues if isinstance(issue, dict) and issue.get('severity') == 'high'])
        
        # Calculate dependency metrics
        total_dependencies = sum(len(deps) for deps in self.dependency_graph.values())
        avg_dependencies = total_dependencies / len(self.dependency_graph) if self.dependency_graph else 0
        
        # Calculate scores
        complexity_score = max(0, 100 - avg_complexity)
        documentation_score = avg_documentation
        dependency_score = max(0, 100 - min(avg_dependencies * 10, 100))
        security_score = max(0, 100 - (critical_issues * 20 + high_issues * 10))
        
        return {
            'complexity_score': {
                'score': complexity_score,
                'grade': self._score_to_grade(complexity_score),
                'average': avg_complexity
            },
            'documentation_score': {
                'score': documentation_score,
                'grade': self._score_to_grade(documentation_score),
                'average': avg_documentation
            },
            'dependency_score': {
                'score': dependency_score,
                'grade': self._score_to_grade(dependency_score),
                'average': avg_dependencies
            },
            'security_score': {
                'score': security_score,
                'grade': self._score_to_grade(security_score),
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'total_issues': len(all_security_issues)
            }
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if not isinstance(score, (int, float)):
            return "F"
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary."""
        console.print("[blue]🔧 Generating analysis summary...[/blue]")
        
        # Language distribution
        language_dist = defaultdict(int)
        for file_data in self.file_analysis.values():
            lang = file_data.get('language', 'unknown')
            language_dist[lang] += 1
        
        # File size distribution
        file_sizes = [file_data.get('size_bytes', 0) for file_data in self.file_analysis.values() if isinstance(file_data.get('size_bytes'), (int, float))]
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        
        # Top files by complexity
        complex_files = []
        for path, data in self.file_analysis.items():
            complexity_score = self._safe_get_score(data.get('complexity_score', 0))
            complex_files.append((path, complexity_score))
        
        complex_files = sorted(complex_files, key=lambda x: x[1], reverse=True)[:10]
        
        # Security summary
        all_security_issues = []
        for file_data in self.file_analysis.values():
            issues = file_data.get('security_issues', [])
            if isinstance(issues, list):
                all_security_issues.extend(issues)
        
        security_by_severity = defaultdict(int)
        for issue in all_security_issues:
            if isinstance(issue, dict):
                security_by_severity[issue.get('severity', 'unknown')] += 1
        
        summary = {
            'file_stats': {
                'total_files': len(self.file_analysis),
                'processed_files': self.processed_files,
                'skipped_files': self.skipped_files,
                'error_files': self.error_files,
                'language_distribution': dict(language_dist),
                'average_file_size': avg_file_size
            },
            'code_metrics': {
                'total_lines': sum(file_data.get('metrics', {}).get('total_lines', 0) for file_data in self.file_analysis.values() if isinstance(file_data.get('metrics'), dict)),
                'total_functions': sum(len(file_data.get('functions', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('functions'), list)),
                'total_classes': sum(len(file_data.get('classes', [])) for file_data in self.file_analysis.values() if isinstance(file_data.get('classes'), list)),
            },
            'complexity_analysis': {
                'most_complex_files': complex_files
            },
            'dependency_analysis': {
                'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
                'files_with_dependencies': len(self.dependency_graph)
            },
            'security_analysis': {
                'total_issues': len(all_security_issues),
                'issues_by_severity': dict(security_by_severity)
            }
        }
        
        return summary
    
    def run_incremental_analysis(self) -> Dict[str, Any]:
        """Run the complete incremental analysis using specialized analyzers."""
        global shutdown_requested
        
        console.print(f"[bold blue]🚀 Starting Enhanced Incremental C/C++ Analysis[/bold blue]")
        console.print(f"[blue]📁 Files to process: {self.total_files}[/blue]")
        console.print(f"[blue]📦 Batch size: {self.batch_size}[/blue]")
        console.print(f"[blue]💾 Memory limit: {self.memory_limit}MB[/blue]")
        console.print(f"[blue]🔧 Analyzers: {len(self.analyzers)} specialized analyzers loaded[/blue]")
        
        start_time = time.time()
        
        # Create file batches
        batches = self.create_file_batches()
        total_batches = len(batches)
        
        console.print(f"[blue]📊 Created {total_batches} batches for processing[/blue]")
        
        # Process batches with progress tracking
        batch_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Processing file batches...", total=total_batches)
            
            for i, batch in enumerate(batches, 1):
                if shutdown_requested:
                    console.print("[yellow]⚠️  Analysis cancelled by user[/yellow]")
                    break
                
                progress.update(main_task, description=f"Processing batch {i}/{total_batches}")
                
                # Process batch
                batch_result = self.analyze_file_batch(batch, i, total_batches)
                batch_results.append(batch_result)
                
                progress.update(main_task, advance=1)
                
                # Memory cleanup after each batch
                if i % 5 == 0:  # Every 5 batches
                    force_garbage_collection()
        
        if not shutdown_requested:
            # Merge all batch results
            self.merge_batch_results(batch_results)
            
            # Calculate health metrics using analyzers
            health_metrics = self.calculate_health_metrics()
            
            # Generate summary
            summary = self.generate_summary()
            
            analysis_time = time.time() - start_time
            
            # Compile final results
            results = {
                'metadata': {
                    'project_name': self.codebase_path.name,
                    'total_files': self.total_files,
                    'processed_files': self.processed_files,
                    'analysis_time': analysis_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'analyzer_version': '3.0',
                    'incremental_analysis': True,
                    'analyzers_used': list(self.analyzers.keys()) if self.analyzers else ['basic']
                },
                'summary': summary,
                'health_metrics': health_metrics,
                'file_cache': list(self.file_analysis.values()),
                'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
                'function_graph': dict(self.function_graph),
                'analysis_metadata': {
                    'analysis_time': analysis_time,
                    'batch_size': self.batch_size,
                    'total_batches': total_batches,
                    'memory_limit': self.memory_limit,
                    'incremental_analysis': True,
                    'analyzer_version': '3.0'
                }
            }
            
            console.print(f"\n[green]✅ Enhanced Incremental Analysis Complete![/green]")
            console.print(f"[blue]⏱️  Total Time: {analysis_time:.1f} seconds[/blue]")
            console.print(f"[blue]📊 Files Processed: {self.processed_files}/{self.total_files}[/blue]")
            console.print(f"[blue]🔧 Analyzers Used: {', '.join(self.analyzers.keys()) if self.analyzers else 'basic'}[/blue]")
            console.print(f"[blue]💾 Final Memory: {get_memory_usage():.1f}MB[/blue]")
            
            return results
        else:
            return {'status': 'cancelled', 'partial_results': batch_results}