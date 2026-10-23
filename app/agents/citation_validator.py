
import re
from dataclasses import dataclass


@dataclass
class CitationValidation:
    valid: bool
    cited_indices: list[int]
    invalid_indices: list[int]
    reason: str


def validate_citations(answer, evidence):
    """
    Validate citation references against
    the retrieved evidence collection.

    This validates citation structure,
    not factual correctness.
    """

    references = re.findall(
        r"\[(\d+)\]",
        answer
    )

    cited = sorted(set(int(x) for x in references))

    invalid = [
        i for i in cited
        if i < 1 or i > len(evidence)
    ]

    if invalid:
        return CitationValidation(
            valid=False,
            cited_indices=cited,
            invalid_indices=invalid,
            reason="Answer contains unknown citations",
        )

    if evidence and not cited:
        return CitationValidation(
            valid=False,
            cited_indices=[],
            invalid_indices=[],
            reason="Answer has no source citations",
        )

    if not evidence:
        return CitationValidation(
            valid=False,
            cited_indices=cited,
            invalid_indices=invalid,
            reason="No evidence available",
        )

    return CitationValidation(
        valid=True,
        cited_indices=cited,
        invalid_indices=[],
        reason="All cited references exist",
    )
