#!/usr/bin/env python3
"""
Tool Wrapper System - Lift Q CLI tools into a higher-level abstraction
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import subprocess
import json

class ToolWrapper(ABC):
    """Base class for wrapping Q CLI tools"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the wrapped tool"""
        pass

class OracleTool(ToolWrapper):
    """Oracle-like search across multiple sources"""
    
    def execute(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Search across multiple sources:
        - Local filesystem
        - Git repositories
        - Web/GitHub
        - Knowledge bases
        """
        results = {
            "query": query,
            "sources": {},
            "found": []
        }
        
        sources = sources or ["local", "git", "web"]
        
        if "local" in sources:
            results["sources"]["local"] = self._search_local(query)
        
        if "git" in sources:
            results["sources"]["git"] = self._search_git(query)
        
        if "web" in sources:
            results["sources"]["web"] = self._search_web(query)
        
        return results
    
    def _search_local(self, query: str) -> List[str]:
        """Search local filesystem"""
        try:
            result = subprocess.run(
                ["find", ".", "-name", f"*{query}*", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip().split('\n') if result.stdout else []
        except Exception as e:
            return [f"Error: {e}"]
    
    def _search_git(self, query: str) -> List[str]:
        """Search git repositories"""
        try:
            result = subprocess.run(
                ["git", "ls-files", f"*{query}*"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip().split('\n') if result.stdout else []
        except Exception as e:
            return [f"Error: {e}"]
    
    def _search_web(self, query: str) -> Dict[str, str]:
        """Search web/GitHub"""
        return {
            "note": "Would use web_search or web_fetch tool here",
            "query": query
        }

class EnhancedGrep(ToolWrapper):
    """Enhanced grep with multiple backends"""
    
    def execute(self, pattern: str, path: str = ".", 
                use_ripgrep: bool = True) -> List[Dict[str, Any]]:
        """Search with fallback to multiple tools"""
        
        if use_ripgrep:
            try:
                return self._ripgrep(pattern, path)
            except:
                pass
        
        return self._fallback_grep(pattern, path)
    
    def _ripgrep(self, pattern: str, path: str) -> List[Dict[str, Any]]:
        """Use ripgrep if available"""
        result = subprocess.run(
            ["rg", "--json", pattern, path],
            capture_output=True,
            text=True
        )
        
        matches = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        matches.append(data)
                except:
                    pass
        
        return matches
    
    def _fallback_grep(self, pattern: str, path: str) -> List[Dict[str, Any]]:
        """Fallback to standard grep"""
        result = subprocess.run(
            ["grep", "-r", "-n", pattern, path],
            capture_output=True,
            text=True
        )
        
        matches = []
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    matches.append({
                        "file": parts[0],
                        "line": parts[1],
                        "content": parts[2]
                    })
        
        return matches

class ToolRegistry:
    """Registry for all wrapped tools"""
    
    def __init__(self):
        self.tools = {
            "oracle": OracleTool(),
            "enhanced_grep": EnhancedGrep(),
        }
    
    def get(self, name: str) -> Optional[ToolWrapper]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def register(self, name: str, tool: ToolWrapper):
        """Register a new tool"""
        self.tools[name] = tool

# Global registry
registry = ToolRegistry()

def oracle(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for oracle search"""
    return registry.get("oracle").execute(query=query, **kwargs)

def enhanced_grep(pattern: str, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function for enhanced grep"""
    return registry.get("enhanced_grep").execute(pattern=pattern, **kwargs)

if __name__ == "__main__":
    # Example usage
    print("Oracle search for 'emoji_report_generator.rs':")
    results = oracle("emoji_report_generator.rs")
    print(json.dumps(results, indent=2))
