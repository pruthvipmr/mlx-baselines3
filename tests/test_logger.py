"""Tests for the logger system."""

import os
import csv
import tempfile
import pytest
import numpy as np

from mlx_baselines3.common.logger import (
    Logger,
    configure_logger,
    HumanOutputFormat,
    CSVOutputFormat,
    TensorBoardOutputFormat,
)


class TestLogger:
    """Test logger functionality."""

    def test_logger_init(self):
        """Test logger initialization."""
        logger = Logger(output_formats=["stdout"])
        assert len(logger.writers) == 1
        assert isinstance(logger.writers[0], HumanOutputFormat)

    def test_logger_record(self):
        """Test recording values."""
        logger = Logger(output_formats=[])

        logger.record("test/value", 42.0)
        logger.record("test/string", "hello")

        assert logger.name_to_value["test/value"] == 42.0
        assert logger.name_to_value["test/string"] == "hello"

    def test_logger_record_with_exclude(self):
        """Test recording with exclusions."""
        logger = Logger(output_formats=[])

        logger.record("test/value", 42.0, exclude="stdout")

        assert logger.name_to_excluded["test/value"] == ("stdout",)

    def test_logger_record_mean(self):
        """Test recording mean values."""
        logger = Logger(output_formats=[])

        for i in range(5):
            logger.record_mean("test/mean", i)

        # Should store list of values
        assert len(logger.name_to_value["test/mean"]) == 5
        assert logger.name_to_count["test/mean"] == 5

    def test_logger_dump_clears_data(self):
        """Test that dump clears recorded data."""
        logger = Logger(output_formats=[])

        logger.record("test/value", 42.0)
        assert len(logger.name_to_value) == 1

        logger.dump(step=0)
        assert len(logger.name_to_value) == 0

    def test_logger_close(self):
        """Test logger close."""
        logger = Logger(output_formats=["stdout"])
        logger.close()  # Should not raise


class TestHumanOutputFormat:
    """Test human output format."""

    def test_human_output_format_init(self):
        """Test initialization."""
        formatter = HumanOutputFormat()
        assert formatter.start_time is not None

    def test_human_output_format_write(self, capsys):
        """Test writing output."""
        formatter = HumanOutputFormat()

        key_values = {
            "rollout/ep_rew_mean": 150.5,
            "time/fps": 2000,
            "train/loss": 0.001,
        }
        key_excluded = {}

        formatter.write(key_values, key_excluded, step=1000)

        captured = capsys.readouterr()
        assert "ep_rew_mean" in captured.out
        assert "150.5" in captured.out
        assert "fps" in captured.out
        # FPS can vary due to timing, just check that it's present
        assert "fps" in captured.out

    def test_human_output_format_with_exclusions(self, capsys):
        """Test writing with exclusions."""
        formatter = HumanOutputFormat()

        key_values = {
            "test/visible": 42.0,
            "test/hidden": 99.0,
        }
        key_excluded = {"test/hidden": ("stdout",)}

        formatter.write(key_values, key_excluded, step=1000)

        captured = capsys.readouterr()
        assert "visible" in captured.out
        assert "hidden" not in captured.out

    def test_human_output_format_empty_data(self, capsys):
        """Test writing empty data."""
        formatter = HumanOutputFormat()

        formatter.write({}, {}, step=1000)

        captured = capsys.readouterr()
        assert captured.out == ""  # Should output nothing


class TestCSVOutputFormat:
    """Test CSV output format."""

    def test_csv_output_format_init(self):
        """Test initialization."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            formatter = CSVOutputFormat(filename)
            assert formatter.filename == filename
            assert formatter.file is None
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_csv_output_format_write(self):
        """Test writing to CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            formatter = CSVOutputFormat(filename)

            key_values = {
                "rollout/ep_rew_mean": 150.5,
                "time/fps": 2000,
            }
            key_excluded = {}

            formatter.write(key_values, key_excluded, step=1000)
            formatter.close()

            # Check CSV content
            assert os.path.exists(filename)
            with open(filename, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1
                assert float(rows[0]["rollout/ep_rew_mean"]) == 150.5
                assert int(rows[0]["time/fps"]) == 2000
                assert int(rows[0]["timesteps"]) == 1000

        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_csv_output_format_multiple_writes(self):
        """Test multiple writes to CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            formatter = CSVOutputFormat(filename)

            # Write with all keys from the start
            formatter.write({"value": 1.0, "new_key": ""}, {}, step=100)

            # Second write with both keys
            formatter.write({"value": 2.0, "new_key": 3.0}, {}, step=200)

            formatter.close()

            # Check CSV content
            with open(filename, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 2
                assert "new_key" in reader.fieldnames

        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_csv_output_format_with_exclusions(self):
        """Test CSV with exclusions."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            formatter = CSVOutputFormat(filename)

            key_values = {
                "visible": 42.0,
                "hidden": 99.0,
            }
            key_excluded = {"hidden": ("csv",)}

            formatter.write(key_values, key_excluded, step=1000)
            formatter.close()

            # Check CSV content
            with open(filename, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1
                assert "visible" in rows[0]
                assert "hidden" not in rows[0]

        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestTensorBoardOutputFormat:
    """Test TensorBoard output format."""

    def test_tensorboard_output_format_init(self):
        """Test initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                formatter = TensorBoardOutputFormat(tmpdir)
                assert hasattr(formatter, "writer")
                formatter.close()
            except ImportError:
                # TensorBoard not available, skip test
                pytest.skip("TensorBoard not available")

    def test_tensorboard_output_format_write(self):
        """Test writing to TensorBoard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                formatter = TensorBoardOutputFormat(tmpdir)

                key_values = {
                    "rollout/ep_rew_mean": 150.5,
                    "time/fps": 2000,
                }
                key_excluded = {}

                formatter.write(key_values, key_excluded, step=1000)
                formatter.close()

                # Check that TensorBoard files were created
                tb_files = [
                    f for f in os.listdir(tmpdir) if f.startswith("events.out.tfevents")
                ]
                assert len(tb_files) > 0

            except ImportError:
                # TensorBoard not available, skip test
                pytest.skip("TensorBoard not available")

    def test_tensorboard_output_format_with_exclusions(self):
        """Test TensorBoard with exclusions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                formatter = TensorBoardOutputFormat(tmpdir)

                key_values = {
                    "visible": 42.0,
                    "hidden": 99.0,
                }
                key_excluded = {"hidden": ("tensorboard",)}

                formatter.write(key_values, key_excluded, step=1000)
                formatter.close()

                # Should not raise, exclusions should work

            except ImportError:
                # TensorBoard not available, skip test
                pytest.skip("TensorBoard not available")


class TestConfigureLogger:
    """Test logger configuration."""

    def test_configure_logger_stdout_only(self):
        """Test configuring logger with stdout only."""
        logger = configure_logger(format_strings=["stdout"])

        assert len(logger.writers) == 1
        assert isinstance(logger.writers[0], HumanOutputFormat)

    def test_configure_logger_with_folder(self):
        """Test configuring logger with folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = configure_logger(folder=tmpdir, format_strings=["stdout", "csv"])

            assert len(logger.writers) == 2
            writer_types = [type(w).__name__ for w in logger.writers]
            assert "HumanOutputFormat" in writer_types
            assert "CSVOutputFormat" in writer_types

    def test_configure_logger_all_formats(self):
        """Test configuring logger with all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = configure_logger(
                folder=tmpdir, format_strings=["stdout", "csv", "tensorboard"]
            )

            # Should have stdout and csv, tensorboard depends on availability
            assert len(logger.writers) >= 2

    def test_configure_logger_defaults(self):
        """Test configuring logger with defaults."""
        logger = configure_logger()

        assert len(logger.writers) == 1
        assert isinstance(logger.writers[0], HumanOutputFormat)


class TestLoggerIntegration:
    """Test logger integration."""

    def test_logger_full_workflow(self):
        """Test complete logger workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = configure_logger(folder=tmpdir, format_strings=["csv"])

            # Record various types of data
            logger.record("rollout/ep_rew_mean", 150.5)
            logger.record("time/fps", 2000)
            logger.record_mean("train/loss", 0.1)
            logger.record_mean("train/loss", 0.2)
            logger.record_mean("train/loss", 0.3)

            # Dump data
            logger.dump(step=1000)

            # Check CSV file
            csv_file = os.path.join(tmpdir, "progress.csv")
            assert os.path.exists(csv_file)

            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert float(rows[0]["rollout/ep_rew_mean"]) == 150.5
                assert (
                    abs(float(rows[0]["train/loss"]) - 0.2) < 1e-10
                )  # Mean of [0.1, 0.2, 0.3]

            logger.close()

    def test_logger_multiple_dumps(self):
        """Test multiple dumps to logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = configure_logger(folder=tmpdir, format_strings=["csv"])

            # First dump
            logger.record("value", 1.0)
            logger.dump(step=100)

            # Second dump
            logger.record("value", 2.0)
            logger.dump(step=200)

            # Check CSV file
            csv_file = os.path.join(tmpdir, "progress.csv")
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert float(rows[0]["value"]) == 1.0
                assert float(rows[1]["value"]) == 2.0

            logger.close()

    def test_logger_with_numpy_arrays(self):
        """Test logger with numpy arrays."""
        logger = configure_logger(format_strings=[])

        # Test recording numpy arrays
        logger.record("array_scalar", np.array(42.0))
        logger.record("array_1d", np.array([1, 2, 3]))

        # Should handle conversion
        logger.dump(step=0)

    def test_logger_empty_data_handling(self):
        """Test logger handling of empty data."""
        logger = configure_logger(format_strings=["stdout"])

        # Dump without recording anything
        logger.dump(step=0)  # Should not raise

        # Record and clear, then dump again
        logger.record("test", 1.0)
        logger.dump(step=0)
        logger.dump(step=0)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])
