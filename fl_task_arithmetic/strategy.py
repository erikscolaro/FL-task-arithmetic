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
from flwr.app import Context
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)
from flwr_datasets import FederatedDataset
import wandb

from fl_task_arithmetic.task import CustomDino, load_server_test_data, test
from utilities.wandb_utils import save_model_to_wandb
import torch
from torch.utils.data import DataLoader

# we override some functions of fedavg to implement checkpointing
# pylint: disable=too-many-instance-attributes
class CustomFedAvg(FedAvg):
    def __init__(self, last_round: int = 0, *args, **kwargs):
        self.last_round = last_round
        super().__init__(*args, **kwargs)

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
                    "round": server_round + self.last_round,
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
                    "round": server_round + self.last_round,
                    **dict(metrics)
                })
       
        return metrics


def get_evaluate_fn(
    run: wandb.Run, model: CustomDino, context: Context
):
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # An epoch is very long, we have to save every round
        # checkpoint_interval  = int(context.run_config["server-checkpoint-interval"])
        # total_round = int(context.run_config["num-server-rounds"])
        print(f"Current server round: {server_round}")
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict=state_dict)

        # Save model every `save_every_round` round and for the last round
        save_model_to_wandb(run, model)

        # Perform evaluation on the model with the given arrays
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = load_server_test_data()

        # TODO: non funziona, non riesce ad ottenere il dtaset da task.

        if dataset is not None:
            testloader = DataLoader(
                dataset=dataset, #type: ignore
                batch_size=context.run_config["client-batch-size"],  # type: ignore[call-operator]
            )
            avg_loss, accuracy = test(model, testloader ,device)

            wandb.log({"server/val_accuracy": accuracy, "server/val_loss": avg_loss, "server_round": server_round})
        else:
            print("Error obtaining the test split from the federated dataset.")

        return MetricRecord()

    return evaluate
