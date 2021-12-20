"""``DaskRunner`` is an ``AbstractRunner`` implementation. It can be
used to distribute execution of ``Node``s in the ``Pipeline`` across
a Dask cluster, taking into account the inter-``Node`` dependencies.
"""
from typing import Any, Dict

from distributed import Client, as_completed
from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import AbstractRunner, run_node


class _DaskDataSet(AbstractDataSet):
    """``_DaskDataSet`` publishes/gets named datasets to/from the Dask
    scheduler."""

    def __init__(self, client: Client, name: str):
        self._client = client
        self._name = name

    def _load(self) -> Any:
        return self._client.get_dataset(self._name)

    def _save(self, data: Any) -> None:
        self._client.publish_dataset(data, name=self._name)

    def _exists(self) -> bool:
        return self._name in self._client.list_datasets()

    def _release(self) -> None:
        self._client.unpublish_dataset(self._name)

    def _describe(self) -> Dict[str, Any]:
        return dict(name=self._name)


class DaskRunner(AbstractRunner):
    """``DaskRunner`` is an ``AbstractRunner`` implementation. It can be
    used to distribute execution of ``Node``s in the ``Pipeline`` across
    a Dask cluster, taking into account the inter-``Node`` dependencies.
    """

    def __init__(self, client_args: Dict[str, Any] = {}, is_async: bool = False):
        """Instantiates the runner by creating a ``distributed.Client``.

        Args:
            client_args: Arguments to pass to the ``distributed.Client``
                constructor.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
        """
        super().__init__(is_async=is_async)
        self._client = Client(**client_args)

    def __del__(self):
        self._client.close()

    def create_default_data_set(self, ds_name: str) -> _DaskDataSet:
        """Factory method for creating the default data set for the runner.

        Args:
            ds_name: Name of the missing data set.

        Returns:
            An instance of ``_DaskDataSet`` to be used for all
            unregistered data sets.
        """
        return _DaskDataSet(self._client, ds_name)

    def _run_node(
        self,
        node: Node,
        catalog: DataCatalog,
        is_async: bool = False,
        run_id: str = None,
        *dependencies: Node,
    ) -> Node:
        """Run a single `Node` with inputs from and outputs to the `catalog`.

        Wraps ``run_node`` to accept the set of ``Node``s that this node
        depends on. When ``dependencies`` are futures, Dask ensures that
        the upstream node futures are completed before running ``node``.

        Args:
            node: The ``Node`` to run.
            catalog: A ``DataCatalog`` containing the node's inputs and outputs.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
            run_id: The id of the pipeline run.
            dependencies: The upstream ``Node``s to allow Dask to handle
                dependency tracking. Their values are not actually used.

        Returns:
            The node argument.

        """
        return run_node(node, catalog, is_async, run_id)

    def _run(
        self, pipeline: Pipeline, catalog: DataCatalog, run_id: str = None
    ) -> None:
        nodes = pipeline.nodes
        node_dependencies = pipeline.node_dependencies
        node_futures = {}

        for node in nodes:
            dependencies = (
                node_futures[dependency] for dependency in node_dependencies[node]
            )
            node_futures[node] = self._client.submit(
                self._run_node, node, catalog, self._is_async, run_id, *dependencies
            )

        for i, (_, node) in enumerate(
            as_completed(node_futures.values(), with_results=True)
        ):
            self._logger.info("Completed node: %s", node.name)
            self._logger.info("Completed %d out of %d tasks", i + 1, len(nodes))
