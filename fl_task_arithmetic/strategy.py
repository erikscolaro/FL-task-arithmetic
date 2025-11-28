from collections.abc import Iterable
from typing import Optional
from logging import INFO, WARNING

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg

from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)
import wandb

# we override some functions of fedavg to implement checkpointing
# pylint: disable=too-many-instance-attributes
class CustomFedAvg(FedAvg):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

            if metrics is not None:
                wandb.log({
                    "round": server_round,
                    **dict(metrics)
                })

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

            if metrics is not None:
                wandb.log({
                    "round": server_round,
                    **dict(metrics)
                })
       
        return metrics
