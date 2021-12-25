"""``DaskRunner`` is an ``AbstractRunner`` implementation. It can be
used to distribute execution of ``Node``s in the ``Pipeline`` across
a Dask cluster, taking into account the inter-``Node`` dependencies.
"""
from collections import Counter
from itertools import chain
from typing import Any, Dict

from distributed import Client, as_completed, worker_client
from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import AbstractRunner, run_node


class _DaskDataSet(AbstractDataSet):
    """``_DaskDataSet`` publishes/gets named datasets to/from the Dask
    scheduler."""

    def __init__(self, name: str):
        self._name = name

    def _load(self) -> Any:
        with worker_client() as client:
            return client.get_dataset(self._name)

    def _save(self, data: Any) -> None:
        with worker_client() as client:
            client.publish_dataset(data, name=self._name, override=True)

    def _exists(self) -> bool:
        return self._name in Client.current().list_datasets()

    def _release(self) -> None:
        Client.current().unpublish_dataset(self._name)

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
        Client(**client_args)

    def __del__(self):
        Client.current().close()

    def create_default_data_set(self, ds_name: str) -> _DaskDataSet:
        """Factory method for creating the default dataset for the runner.

        Args:
            ds_name: Name of the missing dataset.

        Returns:
            An instance of ``_DaskDataSet`` to be used for all
            unregistered datasets.
        """
        return _DaskDataSet(ds_name)

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
        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        node_dependencies = pipeline.node_dependencies
        node_futures = {}

        client = Client.current()
        for node in nodes:
            dependencies = (
                node_futures[dependency] for dependency in node_dependencies[node]
            )
            node_futures[node] = client.submit(
                self._run_node, node, catalog, self._is_async, run_id, *dependencies
            )

        for i, (_, node) in enumerate(
            as_completed(node_futures.values(), with_results=True)
        ):
            self._logger.info("Completed node: %s", node.name)
            self._logger.info("Completed %d out of %d tasks", i + 1, len(nodes))

            # Decrement load counts, and release any datasets we
            # have finished with. This is particularly important
            # for the shared, default datasets we created above.
            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if (
                    load_counts[data_set] < 1
                    and data_set not in pipeline.inputs()
                ):
                    catalog.release(data_set)
            for data_set in node.outputs:
                if (
                    load_counts[data_set] < 1
                    and data_set not in pipeline.outputs()
                ):
                    catalog.release(data_set)

    def run_only_missing(
        self, pipeline: Pipeline, catalog: DataCatalog
    ) -> Dict[str, Any]:
        """Run only the missing outputs from the ``Pipeline`` using the
        datasets provided by ``catalog``, and save results back to the
        same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be
                satisfied.

        Returns:
            Any node outputs that cannot be processed by the
            ``DataCatalog``. These are returned in a dictionary, where
            the keys are defined by the node outputs.
        """
        free_outputs = pipeline.outputs() - set(catalog.list())
        missing = {ds for ds in catalog.list() if not catalog.exists(ds)}
        to_build = free_outputs | missing
        to_rerun = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(
            *to_build
        )

        # We also need any missing datasets that are required to run the
        # `to_rerun` pipeline, including any chains of missing datasets.
        unregistered_ds = pipeline.data_sets() - set(catalog.list())
        # Some of the unregistered datasets could have been published to
        # the scheduler in a previous run, so we need not recreate them.
        missing_unregistered_ds = {
            ds_name for ds_name in unregistered_ds
            if not self.create_default_data_set(ds_name).exists()
        }
        output_to_unregistered = pipeline.only_nodes_with_outputs(
            *missing_unregistered_ds
        )
        input_from_unregistered = to_rerun.inputs() & missing_unregistered_ds
        to_rerun += output_to_unregistered.to_outputs(*input_from_unregistered)

        # We need to add any previously-published, unregistered datasets
        # to the catalog passed to the `run` method, so that it does not
        # think that the `to_rerun` pipeline's inputs are not satisfied.
        catalog = catalog.shallow_copy()
        for ds_name in unregistered_ds - missing_unregistered_ds:
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        return self.run(to_rerun, catalog)
