from typing import Any, Dict

from pretrain.dataset import (
    SubsetLoader,
    SubsetFineWebLoader,
    SubsetFineWebEdu2Loader,
    SubsetFalconLoader,
    SubsetPes2oXLoader,
    SubsetStackV1DedupLoader,
    SubsetStackV2DedupLoader,
    SubsetFineMathLoader,
    SubsetInfiWebMathLoader,
    SubsetPeopleSpeechLoader,    
)

from pretrain.datasets.ids import DatasetId
from transformers import PreTrainedTokenizerBase


class DatasetLoaderFactory:
    @staticmethod
    def get_loader(
        dataset_id: DatasetId,
        dataset_kwargs: Dict[str, Any],
        seed: int,
        sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> SubsetLoader:
        """Loads data samples from the appropriate dataset."""

        match dataset_id:
            case DatasetId.FALCON:
                return SubsetFalconLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.FINEWEB:
                return SubsetFineWebLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.FINEWEB2:
                return SubsetFineWebEdu2Loader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.STACK1:
                return SubsetStackV1DedupLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.STACK2:
                return SubsetStackV2DedupLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.PES2OX:
                return SubsetPes2oXLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.FINEMATH3P:
                return SubsetFineMathLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.WEBMATH3P:
                return SubsetInfiWebMathLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.PPLSPEECH:
                return SubsetPeopleSpeechLoader(
                    random_seed=seed,
                    **dataset_kwargs,
                )
            case _:
                raise ValueError(f"Unknown dataset_id: {dataset_id}")
