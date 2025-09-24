"""
Logger module for MLX Baselines3.

Provides logging functionality for training metrics, including
stdout output, CSV logging, and TensorBoard integration.
"""

import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class HParam:
    """
    Hyperparameter class for TensorBoard logging.
    """

    def __init__(self, value: Any, hparam_type: Optional[str] = None):
        """
        Initialize hyperparameter.

        Args:
            value: The hyperparameter value
            hparam_type: Type of hyperparameter (for TensorBoard)
        """
        self.value = value
        self.hparam_type = hparam_type


class Logger:
    """
    Logger for training metrics and hyperparameters.

    Supports multiple output formats including stdout,
    CSV files, and TensorBoard.
    """

    def __init__(
        self,
        folder: Optional[str] = None,
        output_formats: Optional[List[str]] = None,
    ):
        """
        Initialize the logger.

        Args:
            folder: Folder to save log files
            output_formats: List of output formats ("stdout", "csv", "tensorboard")
        """
        if output_formats is None:
            output_formats = ["stdout"]

        self.folder = folder
        self.output_formats = output_formats
        self.name_to_value = {}
        self.name_to_count = {}
        self.name_to_excluded = {}
        self.level = 1  # INFO level by default

        # Initialize output handlers
        self.writers = []

        if "stdout" in output_formats:
            self.writers.append(HumanOutputFormat())

        if "csv" in output_formats and folder is not None:
            os.makedirs(folder, exist_ok=True)
            self.writers.append(CSVOutputFormat(os.path.join(folder, "progress.csv")))

        if "tensorboard" in output_formats and folder is not None:
            try:
                self.writers.append(TensorBoardOutputFormat(folder))
            except ImportError:
                print("Warning: TensorBoard not available")

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        """
        Log a value with a given key.

        Args:
            key: The key to log
            value: The value to log
            exclude: Format(s) to exclude from logging this key
        """
        if exclude is None:
            exclude = tuple()
        elif isinstance(exclude, str):
            exclude = (exclude,)

        self.name_to_value[key] = value
        self.name_to_excluded[key] = exclude

    def record_mean(
        self, key: str, value: Union[int, float], exclude: Optional[str] = None
    ) -> None:
        """
        Log the mean of a value.

        Args:
            key: The key to log
            value: The value to add to the mean calculation
            exclude: Format to exclude from logging this key
        """
        if exclude is None:
            exclude = tuple()
        elif isinstance(exclude, str):
            exclude = (exclude,)

        if key not in self.name_to_value:
            self.name_to_value[key] = []
            self.name_to_count[key] = 0
            self.name_to_excluded[key] = exclude

        self.name_to_value[key].append(value)
        self.name_to_count[key] += 1

    def dump(self, step: int = 0) -> None:
        """
        Write all logged values to outputs.

        Args:
            step: Current training step/timestep
        """
        # Compute means for averaged values
        key_to_value = {}
        for key, value in self.name_to_value.items():
            if isinstance(value, list):
                key_to_value[key] = np.mean(value) if len(value) > 0 else 0.0
            else:
                key_to_value[key] = value

        # Write to all outputs
        for writer in self.writers:
            writer.write(key_to_value, self.name_to_excluded, step)

        # Clear recorded values
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def close(self) -> None:
        """Close all output writers."""
        for writer in self.writers:
            writer.close()

    def get_dir(self) -> Optional[str]:
        """Get the logging directory."""
        return self.folder

    def set_level(self, level: int) -> None:
        """
        Set the logging level.

        Args:
            level: Logging level (higher = more verbose)
        """
        self.level = level


class OutputFormat:
    """Abstract base class for output formats."""

    def write(
        self, key_values: Dict[str, Any], key_excluded: Dict[str, tuple], step: int
    ) -> None:
        """
        Write key-value pairs.

        Args:
            key_values: Dictionary of key-value pairs to write
            key_excluded: Dictionary mapping keys to excluded formats
            step: Current step/timestep
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the output format."""
        pass


class HumanOutputFormat(OutputFormat):
    """
    Output format for human-readable console output.
    """

    def __init__(self):
        """Initialize human output format."""
        self.start_time = time.time()

    def write(
        self, key_values: Dict[str, Any], key_excluded: Dict[str, tuple], step: int
    ) -> None:
        """Write key-value pairs to stdout."""
        # Filter out keys excluded from stdout
        filtered_kvs = {
            key: value
            for key, value in key_values.items()
            if "stdout" not in key_excluded.get(key, tuple())
        }

        if not filtered_kvs:
            return

        # Create output string
        time_elapsed = time.time() - self.start_time
        output_lines = ["------------------------------------"]
        output_lines.append("| time/               |            |")
        output_lines.append(
            f"|    fps              | {step / max(time_elapsed, 1e-6):10.0f} |"
        )
        output_lines.append(f"|    time_elapsed     | {time_elapsed:10.0f} |")
        output_lines.append(f"|    total_timesteps  | {step:10} |")

        # Group keys by category
        categories = {}
        for key, value in filtered_kvs.items():
            if "/" in key:
                category = key.split("/")[0]
            else:
                category = "misc"

            if category not in categories:
                categories[category] = {}
            categories[category][key] = value

        # Print each category
        for category, items in sorted(categories.items()):
            if category != "time":  # time is already printed above
                output_lines.append(f"| {category}/               |            |")
                for key, value in sorted(items.items()):
                    short_key = key.split("/")[-1] if "/" in key else key
                    if isinstance(value, float):
                        output_lines.append(f"|    {short_key:<12} | {value:10.3f} |")
                    else:
                        output_lines.append(
                            f"|    {short_key:<12} | {str(value):>10} |"
                        )

        output_lines.append("------------------------------------")
        print("\n".join(output_lines))


class CSVOutputFormat(OutputFormat):
    """
    Output format for CSV files.
    """

    def __init__(self, filename: str):
        """
        Initialize CSV output format.

        Args:
            filename: Path to CSV file
        """
        self.filename = filename
        self.file = None
        self.keys = []
        self.header_written = False

    def write(
        self, key_values: Dict[str, Any], key_excluded: Dict[str, tuple], step: int
    ) -> None:
        """Write key-value pairs to CSV file."""
        # Filter out keys excluded from CSV
        filtered_kvs = {
            key: value
            for key, value in key_values.items()
            if "csv" not in key_excluded.get(key, tuple())
        }

        if not filtered_kvs:
            return

        # Add step to values
        filtered_kvs["timesteps"] = step

        # Update keys if new ones found
        extra_keys = set(filtered_kvs.keys()) - set(self.keys)
        if extra_keys:
            self.keys.extend(extra_keys)
            self.keys = sorted(self.keys)
            # Need to recreate file with new headers if already exists
            if self.file is not None:
                self.file.close()
                self.file = None
                self.header_written = False

        if self.file is None:
            self.file = open(self.filename, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
            self.header_written = False

        if not self.header_written:
            self.writer.writeheader()
            self.header_written = True

        self.writer.writerow({key: filtered_kvs.get(key, "") for key in self.keys})
        self.file.flush()

    def close(self) -> None:
        """Close CSV file."""
        if self.file is not None:
            self.file.close()
            self.file = None


class TensorBoardOutputFormat(OutputFormat):
    """
    Output format for TensorBoard logging.
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard output format.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            raise ImportError("TensorBoard logging requires PyTorch: pip install torch")

    def write(
        self, key_values: Dict[str, Any], key_excluded: Dict[str, tuple], step: int
    ) -> None:
        """Write key-value pairs to TensorBoard."""
        # Filter out keys excluded from TensorBoard
        filtered_kvs = {
            key: value
            for key, value in key_values.items()
            if "tensorboard" not in key_excluded.get(key, tuple())
        }

        for key, value in filtered_kvs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                self.writer.add_scalar(key, value.item(), step)

    def close(self) -> None:
        """Close TensorBoard writer."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()


def configure_logger(
    folder: Optional[str] = None, format_strings: Optional[List[str]] = None
) -> Logger:
    """
    Configure logger with specified formats.

    Args:
        folder: Folder to save log files
        format_strings: List of format strings ("stdout", "csv", "tensorboard")

    Returns:
        Configured logger instance
    """
    if format_strings is None:
        format_strings = ["stdout"]

    return Logger(folder=folder, output_formats=format_strings)


# Global logger instance (can be set by algorithms)
current_logger: Optional[Logger] = None


def get_logger() -> Optional[Logger]:
    """Get the current global logger."""
    return current_logger


def set_logger(logger: Logger) -> None:
    """Set the current global logger."""
    global current_logger
    current_logger = logger


def record(key: str, value: Any, exclude: Optional[str] = None) -> None:
    """Record a value using the global logger."""
    if current_logger is not None:
        current_logger.record(key, value, exclude)


def dump(step: int = 0) -> None:
    """Dump logged values using the global logger."""
    if current_logger is not None:
        current_logger.dump(step)
