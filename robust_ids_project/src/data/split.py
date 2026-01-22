from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class DomainSplit:
    train_domains: list[str]
    test_domains: list[str]
    recent_val_domains: list[str]


def _normalize_domain_values(dom: pd.Series) -> pd.Series:
    return dom.astype(str)


def make_domain_split(
    df: pd.DataFrame,
    *,
    domain_col: str,
    test_last_n_domains: int | None = None,
    train_domains: Sequence[str] | None = None,
    test_domains: Sequence[str] | None = None,
    recent_val_n_domains: int = 1,
) -> DomainSplit:
    if domain_col not in df.columns:
        raise KeyError(f"domain_col '{domain_col}' not in dataframe columns.")

    domains = _normalize_domain_values(df[domain_col])
    unique_domains = sorted(domains.dropna().unique().tolist())
    if len(unique_domains) < 2:
        raise ValueError("Need at least 2 unique domains to evaluate distribution shift.")

    if test_domains is not None:
        test_d = [str(x) for x in test_domains]
        if train_domains is None:
            train_d = [d for d in unique_domains if d not in set(test_d)]
        else:
            train_d = [str(x) for x in train_domains]
    else:
        if test_last_n_domains is None:
            raise ValueError("Provide either test_domains or test_last_n_domains.")
        if test_last_n_domains <= 0 or test_last_n_domains >= len(unique_domains):
            raise ValueError("test_last_n_domains must be in [1, n_domains-1].")
        test_d = unique_domains[-test_last_n_domains:]
        train_d = unique_domains[: -test_last_n_domains]

    if len(train_d) == 0 or len(test_d) == 0:
        raise ValueError("Train/test domain split resulted in empty set.")

    recent_val_n_domains = int(recent_val_n_domains)
    if recent_val_n_domains <= 0 or recent_val_n_domains >= len(train_d):
        raise ValueError("recent_val_n_domains must be in [1, n_train_domains-1].")
    recent_val_d = train_d[-recent_val_n_domains:]

    return DomainSplit(train_domains=list(train_d), test_domains=list(test_d), recent_val_domains=list(recent_val_d))


def filter_by_domains(df: pd.DataFrame, *, domain_col: str, domains: Sequence[str]) -> pd.DataFrame:
    dom = _normalize_domain_values(df[domain_col])
    dom_set = set(str(x) for x in domains)
    return df.loc[dom.isin(dom_set)].copy()

