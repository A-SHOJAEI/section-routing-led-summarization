from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionRoleMapper:
    """
    Maps raw section names to a coarse set of roles.

    The plan calls for a robust regex + normalization mapping such as:
    introduction/method/results/discussion/conclusion/other.
    """

    roles: list[str]

    def __post_init__(self) -> None:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be unique")
        if "other" not in self.roles:
            raise ValueError("roles must include 'other'")

    def map_name(self, name: str | None) -> str:
        if not name:
            return "other"
        n = name.strip().lower()
        n = re.sub(r"[^a-z0-9 ]+", " ", n)
        n = re.sub(r"\s+", " ", n).strip()

        # Common headings in scientific papers.
        if re.search(r"\b(introduction|background|motivation|overview)\b", n):
            return "intro"
        if re.search(r"\b(method|methods|methodology|approach|materials|experimental setup|implementation)\b", n):
            return "methods"
        if re.search(r"\b(result|results|experiments?|evaluation|findings)\b", n):
            return "results"
        if re.search(r"\b(discussion|analysis|limitations)\b", n):
            return "discussion"
        if re.search(r"\b(conclusion|conclusions|concluding|future work|conclusion and future work)\b", n):
            return "conclusion"
        return "other"


def role_special_tokens(roles: list[str]) -> list[str]:
    return [f"<sec:{r}>" for r in roles]

