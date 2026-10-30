import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from common.config import (DB_NAME, DEFINITIONS_COLLECTION, ENV_MONGO_URI, MONGO_TIMEOUT_MS,
                           RESULTS_COLLECTION)
from common.errors import ConfigError, DefinitionNotFoundError
from common.results import ResultRecorder

log = logging.getLogger(__name__)


@contextmanager
def open_database(uri: str | None = None) -> Iterator[Database]:
    """Connect using `uri` or the MONGO_URI env var and always close the client afterwards."""
    uri = uri or os.environ.get(ENV_MONGO_URI)
    if not uri:
        raise ConfigError(f"{ENV_MONGO_URI} environment variable is not set")
    client = MongoClient(uri, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
    try:
        yield client[DB_NAME]
    finally:
        client.close()


def fetch_definition(db: Database, test_name: str) -> dict[str, Any]:
    definition = db[DEFINITIONS_COLLECTION].find_one({"test_name": test_name})
    if not definition:
        raise DefinitionNotFoundError(f"Test definition '{test_name}' not found in {DB_NAME}.{DEFINITIONS_COLLECTION}")
    return definition


def save_results(db: Database, recorder: ResultRecorder, definition: dict[str, Any] | None = None) -> int:
    """Insert every recorded result and stamp the definition's `last_run`. Returns the number of results saved."""
    documents = recorder.documents()
    if not documents:
        return 0
    if definition is not None:
        for doc in documents:
            doc["definition_revision"] = definition.get("revision")
    db[RESULTS_COLLECTION].insert_many(documents)

    if definition is not None:
        db[DEFINITIONS_COLLECTION].update_one(
            {"test_name": definition["test_name"]},
            {"$set": {"last_run": recorder.run.started_at}},
        )
    log.info("Saved %d results of run %s to %s.%s", len(documents), recorder.run.run_id, DB_NAME, RESULTS_COLLECTION)
    return len(documents)
