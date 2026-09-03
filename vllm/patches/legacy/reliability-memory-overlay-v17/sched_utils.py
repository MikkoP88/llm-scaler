# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    # llm-scaler fix (#05d): with ignore_eos=True the request processor
    # must strip eos_token_id and must not match stop_token_ids. Guard
    # both paths anyway so a stale/leaked eos can never terminate an
    # ignore_eos stream early, and log loud diagnostics if the invariant
    # is ever violated at runtime.
    if (
        sampling_params.ignore_eos
        and sampling_params.eos_token_id is not None
    ):
        logger.warning(
            "FINISH_DIAG request %s: ignore_eos=True but eos_token_id=%s "
            "is still set (processor invariant violated).",
            request.request_id,
            sampling_params.eos_token_id,
        )
    if (
        not sampling_params.ignore_eos
        and sampling_params.eos_token_id is not None
        and last_token_id == sampling_params.eos_token_id
    ):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        if sampling_params.ignore_eos:
            logger.warning(
                "FINISH_DIAG request %s: ignore_eos=True but token %d "
                "matched stop_token_ids=%s - refusing to stop (#05d "
                "guard). num_output_tokens=%d",
                request.request_id,
                last_token_id,
                sampling_params.stop_token_ids,
                request.num_output_tokens,
            )
        else:
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        if sampling_params.ignore_eos and (
            request.num_output_tokens < request.max_tokens
        ):
            # The #05d signature: an ignore_eos stream ending early on
            # the window cap. Normal (min==max pinned) length caps stay
            # silent to keep logs clean.
            logger.warning(
                "FINISH_DIAG request %s: ignore_eos=True stream "
                "length-capped at num_tokens=%d (max_model_len=%d) with "
                "num_output_tokens=%d < max_tokens=%d.",
                request.request_id,
                request.num_tokens,
                max_model_len,
                request.num_output_tokens,
                request.max_tokens,
            )
        return True

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
