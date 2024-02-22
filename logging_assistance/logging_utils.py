import logging.config


class StartsWithFilter(logging.Filter):
    """
    Filter to only log messages that start with a certain prefix

    The below example can be used in the logging_config.json file:

    "filters": {
    "get_metrics": {
      "()": "logging_helper.StartsWithFilter",
      "prefixes": [
       "ONE_PREFIX_HERE",
       "ANOTHER_PREFIX_THERE"
       ]
    }

    """

    def __init__(self, prefixes: list[str]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        """
        Filter the log records for the given prefix

        :param record: log record
        :return: log record if the message starts with the prefix
        """
        return any(record.getMessage().startswith(prefix) for prefix in self.prefixes)
