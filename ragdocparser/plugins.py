"""
Plugin system for RAGDocParser.
Allows extending functionality through custom plugins.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import inspect
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins supported by the system."""
    PARSER = "parser"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    RETRIEVER = "retriever"
    POSTPROCESSOR = "postprocessor"
    MONITOR = "monitor"
    VECTORDB = "vectordb"

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    config_schema: Optional[Dict[str, Any]] = None
    enabled: bool = True

class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

class ParserPlugin(PluginInterface):
    """Base class for document parser plugins."""
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this plugin can parse the given file."""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path, **kwargs) -> str:
        """Parse the document and return text content."""
        pass

class ChunkerPlugin(PluginInterface):
    """Base class for text chunking plugins."""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text into smaller pieces."""
        pass

class EmbedderPlugin(PluginInterface):
    """Base class for embedding plugins."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

class RetrieverPlugin(PluginInterface):
    """Base class for retrieval plugins."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        pass

class PostProcessorPlugin(PluginInterface):
    """Base class for post-processing plugins."""
    
    @abstractmethod
    def process_results(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Post-process retrieval results."""
        pass

class MonitorPlugin(PluginInterface):
    """Base class for monitoring plugins."""
    
    @abstractmethod
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a monitoring event."""
        pass

class VectorDBPlugin(PluginInterface):
    """Base class for vector database plugins."""
    
    @abstractmethod
    def store_embeddings(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Store embeddings in the vector database."""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass

class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
        self.loaded_plugins: Dict[str, PluginMetadata] = {}
    
    def register_plugin(self, plugin: PluginInterface, config: Dict[str, Any] = None) -> bool:
        """Register a plugin instance."""
        try:
            metadata = plugin.metadata
            
            # Check if plugin is already registered
            if metadata.name in self.plugins:
                logger.warning(f"Plugin {metadata.name} is already registered")
                return False
            
            # Initialize plugin
            plugin.initialize(config or {})
            
            # Register plugin
            self.plugins[metadata.name] = plugin
            self.plugin_types[metadata.plugin_type].append(metadata.name)
            self.loaded_plugins[metadata.name] = metadata
            
            logger.info(f"Registered plugin: {metadata.name} (type: {metadata.plugin_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} is not registered")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            metadata = self.loaded_plugins[plugin_name]
            
            # Cleanup plugin
            plugin.cleanup()
            
            # Remove from registry
            del self.plugins[plugin_name]
            self.plugin_types[metadata.plugin_type].remove(plugin_name)
            del self.loaded_plugins[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type."""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins."""
        return list(self.loaded_plugins.values())
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        metadata = self.loaded_plugins.get(plugin_name)
        return metadata.enabled if metadata else False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.loaded_plugins:
            self.loaded_plugins[plugin_name].enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.loaded_plugins:
            self.loaded_plugins[plugin_name].enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
        return False

class PluginLoader:
    """Loader for plugins from files and directories."""
    
    def __init__(self, registry: PluginRegistry):
        """Initialize the plugin loader."""
        self.registry = registry
    
    def load_plugin_from_file(self, file_path: Path, config: Dict[str, Any] = None) -> bool:
        """Load a plugin from a Python file."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.warning(f"No plugin classes found in {file_path}")
                return False
            
            # Register all plugin classes
            success = True
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    if not self.registry.register_plugin(plugin_instance, config):
                        success = False
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return False
    
    def load_plugins_from_directory(self, directory: Path, config: Dict[str, Any] = None) -> int:
        """Load all plugins from a directory."""
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return 0
        
        loaded_count = 0
        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            if self.load_plugin_from_file(plugin_file, config):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins from {directory}")
        return loaded_count
    
    def load_plugin_from_config(self, config_path: Path) -> bool:
        """Load plugin configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if "plugin_path" not in config:
                logger.error("Plugin configuration missing 'plugin_path'")
                return False
            
            plugin_path = Path(config["plugin_path"])
            plugin_config = config.get("config", {})
            
            if plugin_path.is_file():
                return self.load_plugin_from_file(plugin_path, plugin_config)
            elif plugin_path.is_dir():
                return self.load_plugins_from_directory(plugin_path, plugin_config) > 0
            else:
                logger.error(f"Plugin path does not exist: {plugin_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load plugin config from {config_path}: {e}")
            return False

class PluginManager:
    """Main plugin manager for RAGDocParser."""
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self.hooks: Dict[str, List[Callable]] = {}
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook."""
        results = []
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook callback failed for {hook_name}: {e}")
        return results
    
    def get_enabled_plugins(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all enabled plugins of a specific type."""
        plugins = self.registry.get_plugins_by_type(plugin_type)
        return [p for p in plugins if self.registry.is_plugin_enabled(p.metadata.name)]
    
    def apply_parser_plugins(self, file_path: Path) -> Optional[str]:
        """Apply parser plugins to a file."""
        parser_plugins = self.get_enabled_plugins(PluginType.PARSER)
        
        for plugin in parser_plugins:
            if isinstance(plugin, ParserPlugin) and plugin.can_parse(file_path):
                try:
                    return plugin.parse(file_path)
                except Exception as e:
                    logger.error(f"Parser plugin {plugin.metadata.name} failed: {e}")
        
        return None
    
    def apply_chunker_plugins(self, text: str, **kwargs) -> List[str]:
        """Apply chunker plugins to text."""
        chunker_plugins = self.get_enabled_plugins(PluginType.CHUNKER)
        
        if not chunker_plugins:
            # Default chunking if no plugins
            return [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        # Use the first available chunker plugin
        for plugin in chunker_plugins:
            if isinstance(plugin, ChunkerPlugin):
                try:
                    return plugin.chunk_text(text, **kwargs)
                except Exception as e:
                    logger.error(f"Chunker plugin {plugin.metadata.name} failed: {e}")
        
        return [text]
    
    def apply_postprocessor_plugins(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Apply post-processor plugins to results."""
        postprocessor_plugins = self.get_enabled_plugins(PluginType.POSTPROCESSOR)
        
        processed_results = results
        for plugin in postprocessor_plugins:
            if isinstance(plugin, PostProcessorPlugin):
                try:
                    processed_results = plugin.process_results(processed_results, **kwargs)
                except Exception as e:
                    logger.error(f"Post-processor plugin {plugin.metadata.name} failed: {e}")
        
        return processed_results
    
    def cleanup_all_plugins(self) -> None:
        """Cleanup all registered plugins."""
        for plugin_name in list(self.registry.plugins.keys()):
            self.registry.unregister_plugin(plugin_name)
    
    def export_plugin_config(self, file_path: Path) -> None:
        """Export current plugin configuration."""
        config = {
            "plugins": [
                {
                    "name": metadata.name,
                    "type": metadata.plugin_type.value,
                    "version": metadata.version,
                    "enabled": metadata.enabled,
                    "description": metadata.description
                }
                for metadata in self.registry.list_plugins()
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Plugin configuration exported to {file_path}")

# Global plugin manager
_global_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager."""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager

def register_plugin(plugin: PluginInterface, config: Dict[str, Any] = None) -> bool:
    """Convenience function to register a plugin."""
    manager = get_plugin_manager()
    return manager.registry.register_plugin(plugin, config)

def load_plugins_from_directory(directory: Union[str, Path], config: Dict[str, Any] = None) -> int:
    """Convenience function to load plugins from a directory."""
    manager = get_plugin_manager()
    return manager.loader.load_plugins_from_directory(Path(directory), config) 