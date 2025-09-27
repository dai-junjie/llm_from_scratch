import pathlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import mimetypes
import json
import yaml


class AgentFileReader:
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
    def read_text(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        try:
            path = self._resolve_path(file_path)
            if not self._is_safe_path(path):
                self.logger.warning(f"Unsafe path access attempted: {path}")
                return None
            return path.read_text(encoding=encoding)
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def read_lines(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> List[str]:
        content = self.read_text(file_path, encoding)
        return content.splitlines() if content else []
        
    def read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        try:
            content = self.read_text(file_path)
            if content:
                return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
        return None
        
    def read_yaml(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        try:
            content = self.read_text(file_path)
            if content:
                return yaml.safe_load(content)
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parse error in {file_path}: {e}")
        return None
        
    def read_binary(self, file_path: Union[str, Path]) -> Optional[bytes]:
        try:
            path = self._resolve_path(file_path)
            if not self._is_safe_path(path):
                self.logger.warning(f"Unsafe path access attempted: {path}")
                return None
            return path.read_bytes()
        except Exception as e:
            self.logger.error(f"Error reading binary file {file_path}: {e}")
            return None
            
    def list_files(self, directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
        try:
            dir_path = self._resolve_path(directory)
            if not self._is_safe_path(dir_path) or not dir_path.is_dir():
                return []
            
            if recursive:
                return list(dir_path.rglob(pattern))
            else:
                return list(dir_path.glob(pattern))
        except Exception as e:
            self.logger.error(f"Error listing files in {directory}: {e}")
            return []
            
    def list_by_extension(self, directory: Union[str, Path], extensions: List[str], recursive: bool = False) -> List[Path]:
        all_files = []
        for ext in extensions:
            pattern = f"*.{ext.lstrip('.')}"
            all_files.extend(self.list_files(directory, pattern, recursive))
        return all_files
        
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        try:
            path = self._resolve_path(file_path)
            if not path.exists():
                return None
                
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(str(path))
            
            return {
                'name': path.name,
                'path': str(path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'suffix': path.suffix,
                'mime_type': mime_type,
                'parent': str(path.parent)
            }
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {e}")
            return None
            
    def find_files_by_content(self, directory: Union[str, Path], search_text: str, 
                            file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        results = []
        try:
            if file_extensions:
                files = self.list_by_extension(directory, file_extensions, recursive=True)
            else:
                files = self.list_files(directory, "*", recursive=True)
                
            for file_path in files:
                if file_path.is_file():
                    content = self.read_text(file_path)
                    if content and search_text in content:
                        results.append({
                            'file': str(file_path),
                            'matches': content.count(search_text)
                        })
        except Exception as e:
            self.logger.error(f"Error searching files: {e}")
            
        return results
        
    def read_directory_tree(self, directory: Union[str, Path], max_depth: int = 3) -> Dict[str, Any]:
        def _build_tree(path: Path, current_depth: int) -> Dict[str, Any]:
            if current_depth > max_depth:
                return {}
                
            tree = {
                'name': path.name,
                'path': str(path),
                'is_dir': path.is_dir(),
                'children': []
            }
            
            if path.is_dir():
                try:
                    for child in sorted(path.iterdir()):
                        if self._is_safe_path(child):
                            tree['children'].append(_build_tree(child, current_depth + 1))
                except PermissionError:
                    pass
                    
            return tree
            
        try:
            dir_path = self._resolve_path(directory)
            return _build_tree(dir_path, 0)
        except Exception as e:
            self.logger.error(f"Error building directory tree: {e}")
            return {}
            
    def batch_read_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Optional[str]]:
        results = {}
        for file_path in file_paths:
            results[str(file_path)] = self.read_text(file_path)
        return results
        
    def filter_readable_files(self, file_paths: List[Union[str, Path]]) -> List[Path]:
        readable = []
        for file_path in file_paths:
            path = self._resolve_path(file_path)
            if path.exists() and path.is_file() and self._is_readable(path):
                readable.append(path)
        return readable
        
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        return path.resolve()
        
    def _is_safe_path(self, path: Path) -> bool:
        try:
            resolved_path = path.resolve()
            base_resolved = self.base_path.resolve()
            return str(resolved_path).startswith(str(base_resolved))
        except Exception:
            return False
            
    def _is_readable(self, path: Path) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size < 100 * 1024 * 1024  # 100MB limit
        except Exception:
            return False


def create_agent_file_reader(base_path: Optional[Union[str, Path]] = None) -> AgentFileReader:
    return AgentFileReader(base_path)


def read_file(file_path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    reader = create_agent_file_reader(base_path)
    return reader.read_text(file_path)


def list_project_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    reader = create_agent_file_reader()
    if extensions:
        return reader.list_by_extension(directory, extensions, recursive=True)
    return reader.list_files(directory, "*", recursive=True)