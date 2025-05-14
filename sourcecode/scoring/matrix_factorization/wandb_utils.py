"""
This file creates a wrapper around the Weights & Biases API.

It is used to log metrics to Weights & Biases hosted on Twitter.

To use it, set at least the following environment variables:

WANDB_ENABLED=true
WANDB_PROJECT=test-project
WANDB_ENTITY=birdwatch-service
WANDB_API_KEY=...

Optionally, you can set the following environment variables to configure the run:

WANDB_GROUP=test-group
WANDB_NAME=test-name
"""

import getpass
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import wandb as _wandb


if sys.version_info >= (3, 8):
  from typing import Literal, TypedDict  # pylint: disable=no-name-in-module
else:
  from typing_extensions import Literal, TypedDict

logger = logging.getLogger("birdwatch.wandb_utils")
logger.setLevel(logging.INFO)


# See: https://docs.wandb.ai/ref/python/init or wandb's source code
class WandbInitArgs(TypedDict, total=False):  # total=False makes all keys optional
  project: str
  name: str
  group: str
  job_type: str
  tags: List[str]
  entity: str
  config: Dict[str, Any]
  mode: Literal["online", "offline", "disabled"]
  dir: str  # noqa: A003
  id: str  # noqa: A003
  notes: str
  settings: Dict[str, Any]
  reinit: bool
  resume: Union[bool, str, Literal["allow", "must", "never", "auto"]]
  anonymous: Literal["allow", "must", "never"]
  force: bool
  sync_tensorboard: bool
  monitor_gym: bool
  save_code: bool


# Our custom config, restricted to valid wandb.init() args
WANDB_CONFIG: WandbInitArgs = {
  "project": os.getenv("WANDB_PROJECT", "community-notes-scoring"),
  "entity": os.getenv("WANDB_ENTITY", "birdwatch-service"),
  "name": os.getenv("WANDB_NAME", "full-pipeline"),
  "group": os.getenv("WANDB_GROUP", "unknown-group"),
  "settings": _wandb.Settings(
    _file_stream_retry_max=15,
  ),
}


class WandbProxy:
  def __init__(self):
    self._run: Optional[_wandb.wandb_run.Run] = None

  @property
  def _enabled(self) -> bool:
    """Dynamically check if Weights & Biases is enabled from environment variable."""
    return os.getenv("WANDB_ENABLED", "false").lower() == "true"

  def _ensure_init(self) -> None:
    host = os.getenv("WANDB_HOST", "https://wandb.twitter.biz/")
    if self._run is None:
      try:
        if self._enabled:
          logger.info("Logging into W&B and initializing run")
          _wandb.login(host=host, key=self._get_wandb_key())
          self._run = _wandb.init(**WANDB_CONFIG)
        else:
          logger.info("W&B is disabled")
          self._run = _wandb.init(mode="disabled")
      except Exception as e:
        logger.error(f"W&B initialization failed: {e}")
        self._run = _wandb.init(mode="disabled")

  def __getattr__(self, name: str) -> Any:
    self._ensure_init()
    # Special case for module-level classes like Histogram, Table, etc.
    if name in dir(_wandb) and not hasattr(self._run, name):
      return getattr(_wandb, name)
    # Otherwise, delegate to the run object if initialized
    return getattr(self._run, name) if self._run else getattr(_wandb, name)

  def _get_wandb_key(self) -> str:
    """Takes a config and loads the wandb key."""

    if os.getenv("WANDB_API_KEY"):
      return os.getenv("WANDB_API_KEY", "")

    key_path = f"/var/lib/tss/keys/{getpass.getuser()}/onpremwandb.key"

    with open(key_path, "r") as key_file:
      wandb_key = key_file.read().strip()

    return wandb_key

  def log(self, *args: Any, **kwargs: Any) -> None:
    self._ensure_init()
    if self._run:
      self._run.log(*args, **kwargs)

  def reinitialize(self, run_name, config=None):
    logger.info("Reinitializing W&B run")
    if self._run is not None:
      self._run.finish()
    kwargs = {"name": run_name}
    if config is not None:
      kwargs["config"] = config
    set_wandb_config(**kwargs)
    self._run = None  # Force reinitialization on next log
    self._ensure_init()

  def finish(self) -> None:
    if self._run is not None:
      logger.info("Finishing W&B run")
      self._run.finish()
      self._run = None


wandb = WandbProxy()


VALID_INIT_KEYS = set(WandbInitArgs.__annotations__.keys())


def set_wandb_config(**kwargs: Any) -> None:
  if wandb._run is not None:
    print("Warning: W&B already initialized, config update ignored.")
    return

  # Validate keys at runtime
  invalid_keys = set(kwargs.keys()) - VALID_INIT_KEYS
  if invalid_keys:
    raise ValueError(
      f"Invalid wandb.init() arguments: {invalid_keys}. " f"Allowed keys: {VALID_INIT_KEYS}"
    )

  # Update with type safety
  WANDB_CONFIG.update(kwargs)  # type: ignore  # mypy will still enforce types
