"""Config system: YAML with inheritance, strict validation and run snapshots."""

from .checker import COMMAND_REQUIREMENTS, RULES, rule
from .config import Config, available_configs, configs_root, resolve_config_path
from .errors import REFUSAL_KINDS, ConfigError, Issue, Refusal
from .loader import BASE_KEY, DELETE_KEY, MAX_BASE_HOPS, TYPE_KEY, parse_yaml
from .snapshot import git_info, write_snapshot

__all__ = [
    "BASE_KEY",
    "COMMAND_REQUIREMENTS",
    "DELETE_KEY",
    "MAX_BASE_HOPS",
    "REFUSAL_KINDS",
    "RULES",
    "TYPE_KEY",
    "Config",
    "ConfigError",
    "Issue",
    "Refusal",
    "available_configs",
    "configs_root",
    "git_info",
    "parse_yaml",
    "resolve_config_path",
    "rule",
    "write_snapshot",
]
