import torch
import torch.nn.functional as F
from attr.validators import max_len


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["raw_length"] = torch.tensor(
        [elem["audio"].shape[1] for elem in dataset_items]
    )
    max_length = torch.max(result_batch["raw_length"])
    # used 200 because encoder shrinks sequence by 5*5*4*2=200 times
    max_length += (200 - max_length % 200) % 200

    result_batch["audio"] = torch.stack(
        [
            F.pad(elem["audio"], (0, max_length - elem["audio"].shape[1]), value=0.0)
            for elem in dataset_items
        ],
        dim=0,
    )
    result_batch["sample_rate"] = torch.tensor(
        [elem["sample_rate"] for elem in dataset_items]
    )

    return result_batch
