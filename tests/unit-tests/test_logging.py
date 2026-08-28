import logging

from mlonmcu import logging as mlonmcu_logging


def test_get_formatter_supports_minimal_and_detailed_formats():
    record = logging.LogRecord("test", logging.INFO, __file__, 10, "hello", (), None)
    assert mlonmcu_logging.get_formatter(minimal=True).format(record) == "INFO - hello"
    detailed = mlonmcu_logging.get_formatter().format(record)
    assert "test_logging.py:10" in detailed
    assert detailed.endswith("INFO - hello")


def test_get_logger_initializes_a_single_stream_handler(monkeypatch):
    logger = logging.getLogger("mlonmcu")
    old_handlers = logger.handlers[:]
    logger.handlers.clear()
    monkeypatch.setattr(mlonmcu_logging, "initialized", False)
    try:
        assert mlonmcu_logging.get_logger() is logger
        assert len(logger.handlers) == 1
        assert mlonmcu_logging.initialized is True
        mlonmcu_logging.get_logger()
        assert len(logger.handlers) == 1
    finally:
        logger.handlers[:] = old_handlers


def test_set_log_level_updates_logger_and_stream_handlers():
    logger = mlonmcu_logging.get_logger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        mlonmcu_logging.set_log_level(logging.ERROR)
        assert logger.level == logging.ERROR
        assert handler.level == logging.ERROR
    finally:
        logger.removeHandler(handler)


def test_set_log_file_replaces_existing_file_handler(tmp_path):
    logger = mlonmcu_logging.get_logger()
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    mlonmcu_logging.set_log_file(first)
    mlonmcu_logging.set_log_file(second, level=logging.INFO, rotate=True)
    handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    try:
        assert len(handlers) == 1
        assert handlers[0].baseFilename == str(second)
        assert handlers[0].level == logging.INFO
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
