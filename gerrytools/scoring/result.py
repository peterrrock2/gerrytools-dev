"""Semantic results returned by plan scoring."""

from __future__ import annotations

import hashlib
import inspect
import json
import numbers
import os
import re
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

_Dtype: TypeAlias = Literal["bool", "float", "int"]
_Shape = Literal["district", "plan", "region"]
_ReadOperation = Literal["frames", "raw", "read"]
_PANDAS_DTYPES = {"bool": "bool", "float": "float64", "int": "int64"}
_DTYPE_WIDTHS = {"bool": 1, "float": 8, "int": 8}
_EvaluationValue: TypeAlias = bool | float | int | pd.Series | pd.DataFrame
_PREFIX_COLUMNS = ("sample_offset", "repetitions", "accepted_index")
_RESULT_NAME = re.compile(r"[A-Za-z0-9_.-]+")
_WARNING_BYTES = 2 * 1024**3
_ERROR_BYTES = 8 * 1024**3
_FOOTER_EXPANSION = 16


class EvaluationMemoryError(MemoryError):
    """A result read rejected before its predicted peak exceeds the safety limit."""

    def __init__(self, message: str, estimated_bytes: int, limit_bytes: int) -> None:
        super().__init__(message)
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes


def is_valid_metric_name(name: object) -> bool:
    """Whether ``name`` is safe as a metric instance and output path component."""
    return (
        isinstance(name, str)
        and name not in {".", ".."}
        and name.casefold() != "manifest.json"
        and _RESULT_NAME.fullmatch(name) is not None
    )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Counts produced by a batch or streaming evaluation.

    ``samples`` is the number of plan occurrences and includes streaming frame repetitions.
    ``accepted`` is the number of result rows; it equals ``samples`` for batch evaluation. Unique
    counts ignore district labels and are ``None`` when tracking was not requested.
    """

    samples: int
    accepted: int
    unique_plans: int | None = None
    unique_districts: int | None = None


@dataclass(frozen=True, slots=True)
class _MetricResult:
    """One logical result in batched canonical axis order."""

    values: NDArray[np.float64]
    shape: _Shape
    columns: tuple[Hashable, ...]
    districts: tuple[Hashable, ...]
    dtypes: tuple[_Dtype, ...]
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None

    def __post_init__(self) -> None:
        if self.shape == "region":
            expected = (len(self.columns), len(self.regions), len(self.districts))
            valid = (
                self.region_name is not None
                and self.values.ndim == 4
                and self.values.shape[1:] == expected
            )
        elif self.region_name is not None or self.regions:
            valid = False
        elif self.shape == "district":
            expected = (len(self.columns), len(self.districts))
            valid = self.values.ndim == 3 and self.values.shape[1:] == expected
        else:
            valid = (
                not self.districts
                and self.values.ndim == 2
                and self.values.shape[1] == len(self.columns)
            )
        if not valid or len(self.dtypes) != len(self.columns):
            raise ValueError("metric result values and axis metadata do not agree")
        base_array = self.values
        while True:
            parent = base_array.base
            if not isinstance(parent, np.ndarray):
                break
            base_array = parent
        base_array.setflags(write=False)
        self.values.setflags(write=False)


def _index(values: Iterable[Hashable], name: str) -> pd.Index:
    labels = tuple(values)
    array = np.empty(len(labels), dtype=object)
    array[:] = labels
    return pd.Index(array, name=name)


def _metric(results: Mapping[str, _MetricResult], name: str) -> _MetricResult:
    try:
        return results[name]
    except KeyError:
        available = ", ".join(repr(key) for key in results)
        raise KeyError(f"unknown metric {name!r}; available: {available}") from None


def _scalar(value: np.float64, dtype: _Dtype) -> bool | float | int:
    if dtype == "bool":
        return bool(value)
    if dtype == "int":
        return int(value)
    return float(value)


def _cast_columns(
    frame: pd.DataFrame,
    columns: Iterable[Hashable],
    dtypes: Iterable[_Dtype],
) -> pd.DataFrame:
    return frame.astype(
        {column: _PANDAS_DTYPES[dtype] for column, dtype in zip(columns, dtypes, strict=True)}
    )


def _writable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy frozen result values before wrapping them in mutable pandas objects."""
    return np.array(values)


class _Evaluation(Mapping[str, _EvaluationValue]):
    """Read-only logical metric mapping shared by plan and ensemble results."""

    _single: bool

    def __init__(self, results: Mapping[str, _MetricResult]) -> None:
        self._results = dict(results)

    @property
    def metrics(self) -> tuple[str, ...]:
        """Logical metric names in registration order."""
        return tuple(self._results)

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def array(self, name: str) -> NDArray[np.float64]:
        """Return an immutable array ordered like the axes of ``result[name]``."""
        values = _metric(self._results, name).values
        return (values[0] if self._single else values).view()


class PlanEvalResult(_Evaluation):
    """Semantic metric values for one plan."""

    _single = True

    def __init__(self, results: Mapping[str, _MetricResult]) -> None:
        if any(len(result.values) != 1 for result in results.values()):
            raise ValueError("a plan evaluation requires exactly one row per metric")
        super().__init__(results)

    def __getitem__(self, name: str) -> _EvaluationValue:
        result = _metric(self._results, name)
        values = result.values[0]
        columns = _index(result.columns, "metric")
        if result.shape == "region":
            assert result.region_name is not None
            regions = _index(result.regions, result.region_name)
            districts = _index(result.districts, "district")
            result_columns = pd.MultiIndex.from_product(
                (columns, districts),
                names=("metric", "district"),
            )
            frame = pd.DataFrame(
                values.transpose(1, 0, 2).reshape(len(regions), len(result_columns)),
                index=regions,
                columns=result_columns,
            )
            dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
            return _cast_columns(frame, result_columns, dtypes)

        if result.shape == "plan":
            if len(result.columns) == 1:
                return _scalar(values[0], result.dtypes[0])
            typed_values = [
                _scalar(value, dtype) for value, dtype in zip(values, result.dtypes, strict=True)
            ]
            return pd.Series(typed_values, index=columns, name=name)

        districts = _index(result.districts, "district")
        if len(result.columns) == 1:
            return pd.Series(
                _writable(values[0]),
                index=districts,
                name=name,
                dtype=_PANDAS_DTYPES[result.dtypes[0]],
            )
        frame = pd.DataFrame(values.T, index=districts, columns=columns)
        return _cast_columns(frame, result.columns, result.dtypes)


class EnsembleEvalResult(_Evaluation):
    """Semantic metric values for an ordered collection of plans."""

    _single = False

    def __init__(
        self,
        results: Mapping[str, _MetricResult],
        sample_ids: Iterable[Hashable] | None = None,
        *,
        summary: EvaluationSummary | None = None,
        _sample_name: str = "sample",
        _sample_index: pd.Index | None = None,
    ) -> None:
        row_counts = {len(result.values) for result in results.values()}
        if not row_counts:
            raise ValueError("ensemble evaluation requires at least one metric result")
        if len(row_counts) != 1:
            raise ValueError("ensemble metrics must contain the same number of plans")
        count = row_counts.pop()
        if summary is None:
            summary = EvaluationSummary(count, count)
        elif summary.samples != count or summary.accepted != count:
            raise ValueError("batch evaluation summary must match its number of plans")
        self.summary = summary
        if _sample_index is not None:
            if sample_ids is not None or len(_sample_index) != count:
                raise ValueError("private sample index must match the result rows")
            self._samples = _sample_index
        elif sample_ids is None:
            self._samples = pd.RangeIndex(count, name=_sample_name)
        else:
            self._samples = _index(sample_ids, _sample_name)
            if len(self._samples) != count:
                raise ValueError(f"sample_ids has {len(self._samples)} values; expected {count}")
            try:
                has_duplicates = self._samples.has_duplicates
            except TypeError:
                raise ValueError("sample_ids must contain unique hashable values") from None
            if has_duplicates:
                raise ValueError("sample_ids must contain unique hashable values")
        super().__init__(results)

    def __getitem__(self, name: str) -> _EvaluationValue:
        result = _metric(self._results, name)
        values = result.values
        if result.shape == "region":
            assert result.region_name is not None
            regions = _index(result.regions, result.region_name)
            districts = _index(result.districts, "district")
            row_index = pd.MultiIndex.from_product(
                (self._samples, regions),
                names=(self._samples.name, result.region_name),
            )
            columns = pd.MultiIndex.from_product(
                (_index(result.columns, "metric"), districts),
                names=("metric", "district"),
            )
            frame = pd.DataFrame(
                values.transpose(0, 2, 1, 3).reshape(len(row_index), len(columns)),
                index=row_index,
                columns=columns,
            )
            dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
            return _cast_columns(frame, columns, dtypes)

        if result.shape == "plan":
            if len(result.columns) == 1:
                return pd.Series(
                    _writable(values[:, 0]),
                    index=self._samples,
                    name=name,
                    dtype=_PANDAS_DTYPES[result.dtypes[0]],
                )
            columns = _index(result.columns, "metric")
            frame = pd.DataFrame(values, index=self._samples, columns=columns)
            return _cast_columns(frame, result.columns, result.dtypes)

        districts = _index(result.districts, "district")
        if len(result.columns) == 1:
            return pd.DataFrame(
                _writable(values[:, 0, :]),
                index=self._samples,
                columns=districts,
                dtype=_PANDAS_DTYPES[result.dtypes[0]],
            )
        columns = pd.MultiIndex.from_product(
            (_index(result.columns, "metric"), districts),
            names=("metric", "district"),
        )
        frame = pd.DataFrame(values.reshape(len(values), -1), index=self._samples, columns=columns)
        dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
        return _cast_columns(frame, columns, dtypes)


@dataclass(frozen=True, slots=True)
class _RunMetric:
    name: str
    shape: _Shape
    table: Path
    subkeys: tuple[str, ...]
    columns: tuple[str, ...]
    dtypes: tuple[_Dtype, ...]
    table_size: int
    table_sha256: str
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None


class EvaluationRun:
    """A completed streamed scoring run that can reconstruct logical metric results."""

    def __init__(
        self,
        path: Path,
        summary: EvaluationSummary,
        districts: tuple[int, ...],
        metrics: tuple[_RunMetric, ...],
    ) -> None:
        self.path = path
        self.summary = summary
        self._districts = districts
        self._metric_metadata = {metric.name: metric for metric in metrics}
        self._frames: pd.DataFrame | None = None

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> "EvaluationRun":
        """Open and validate a successfully published evaluation run."""
        run_path = Path(path)
        manifest_path = run_path / "manifest.json"
        try:
            with manifest_path.open() as file:
                manifest = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluation manifest JSON: {error}") from error
        summary, districts, metrics = _parse_manifest(run_path, manifest)
        return cls(run_path, summary, districts, metrics)

    @property
    def metrics(self) -> tuple[str, ...]:
        """Logical metric names in registration order."""
        return tuple(self._metric_metadata)

    @property
    def frames(self) -> pd.DataFrame:
        """Accepted stream frames indexed by ``accepted``."""
        if self._frames is None:
            metric = next(iter(self._metric_metadata.values()))
            table = _read_eager_table(
                metric,
                self.summary,
                self._districts,
                "frames",
                columns=list(_PREFIX_COLUMNS),
                allow_large=False,
            )
            self._frames = _validated_frames(table, self.summary)
        frames = self._frames
        assert frames is not None
        return frames.copy()

    def raw(self, name: str, *, allow_large: bool = False) -> pd.DataFrame:
        """Read one metric's physical Parquet table."""
        _validate_bool(allow_large, "allow_large")
        metric = _run_metric(self._metric_metadata, name)
        table = _read_eager_table(
            metric,
            self.summary,
            self._districts,
            "raw",
            allow_large=allow_large,
        )
        self._cache_or_compare_frames(metric, table)
        return table

    def read(
        self,
        name: str,
        *,
        expand_repetitions: bool = False,
        allow_large: bool = False,
    ) -> _EvaluationValue:
        """Read one logical metric, optionally expanding repeated stream frames."""
        _validate_bool(expand_repetitions, "expand_repetitions")
        _validate_bool(allow_large, "allow_large")
        metric = _run_metric(self._metric_metadata, name)
        table = _read_eager_table(
            metric,
            self.summary,
            self._districts,
            "read",
            expand_repetitions=expand_repetitions,
            allow_large=allow_large,
        )
        self._cache_or_compare_frames(metric, table)
        values = _physical_values(table)
        if expand_repetitions:
            repetitions = table["repetitions"].to_numpy(dtype=np.intp)
            values = np.repeat(values, repetitions, axis=0)
        sample_name = "sample" if expand_repetitions else "accepted"
        index = pd.RangeIndex(len(values), name=sample_name)
        return _semantic_value(
            metric.name,
            metric,
            self._districts,
            values,
            index,
        )

    def iter_raw_batches(
        self,
        name: str,
        *,
        batch_size: int = 1_024,
        allow_large: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded physical-table batches for one metric."""
        batch_size = _validate_batch_options(batch_size, allow_large)
        metric = _run_metric(self._metric_metadata, name)
        expected_offset = 0
        expected_accepted = 0
        with _open_metric_reader(
            metric,
            self.summary,
            self._districts,
            "raw",
            allow_large=allow_large,
            batch_size=batch_size,
        ) as parquet:
            columns = list(_PREFIX_COLUMNS) + _value_columns(metric, self._districts)
            for record_batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
                use_threads=False,
            ):
                table = record_batch.to_pandas(use_threads=False)
                del record_batch
                _, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                )
                table.index = pd.RangeIndex(
                    expected_accepted - len(table),
                    expected_accepted,
                )
                output = table
                del table
                yield output
                del output
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def iter_frame_batches(
        self,
        *,
        batch_size: int = 1_024,
        allow_large: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded accepted-frame batches without populating the eager cache."""
        batch_size = _validate_batch_options(batch_size, allow_large)
        metric = next(iter(self._metric_metadata.values()))
        expected_offset = 0
        expected_accepted = 0
        with _open_metric_reader(
            metric,
            self.summary,
            self._districts,
            "frames",
            allow_large=allow_large,
            batch_size=batch_size,
        ) as parquet:
            for record_batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=list(_PREFIX_COLUMNS),
                use_threads=False,
            ):
                table = record_batch.to_pandas(use_threads=False)
                del record_batch
                output, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                    return_frame=True,
                )
                assert output is not None
                del table
                yield output
                del output
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1_024,
        expand_repetitions: bool = False,
        allow_large: bool = False,
    ) -> Iterator[_EvaluationValue]:
        """Yield bounded semantic batches for one metric."""
        batch_size = _validate_batch_options(batch_size, allow_large)
        _validate_bool(expand_repetitions, "expand_repetitions")
        metric = _run_metric(self._metric_metadata, name)
        expected_offset = 0
        expected_accepted = 0
        sample_start = 0
        with _open_metric_reader(
            metric,
            self.summary,
            self._districts,
            "read",
            expand_repetitions=expand_repetitions,
            allow_large=allow_large,
            batch_size=batch_size,
        ) as parquet:
            columns = list(_PREFIX_COLUMNS) + _value_columns(metric, self._districts)
            for record_batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
                use_threads=False,
            ):
                table = record_batch.to_pandas(use_threads=False)
                del record_batch
                accepted_start = expected_accepted
                _, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                )
                values = _physical_values(table)
                if not expand_repetitions:
                    index = pd.RangeIndex(
                        accepted_start,
                        expected_accepted,
                        name="accepted",
                    )
                    output = _semantic_value(name, metric, self._districts, values, index)
                    del table, values
                    yield output
                    del output
                    continue

                repetitions = table["repetitions"].to_numpy(dtype=np.intp, copy=True)
                del table
                # The estimate keeps this accepted-value buffer live while repetitions are split.
                row = 0
                remaining = int(repetitions[0]) if len(repetitions) else 0
                while row < len(values):
                    logical_rows = min(batch_size, expected_offset - sample_start)
                    buffer = np.empty((logical_rows, values.shape[1]), dtype=np.float64)
                    filled = 0
                    while filled < logical_rows:
                        take = min(remaining, logical_rows - filled)
                        buffer[filled : filled + take] = values[row]
                        filled += take
                        remaining -= take
                        if remaining == 0:
                            row += 1
                            if row < len(values):
                                remaining = int(repetitions[row])
                    index = pd.RangeIndex(
                        sample_start,
                        sample_start + logical_rows,
                        name="sample",
                    )
                    sample_start += logical_rows
                    output = _semantic_value(name, metric, self._districts, buffer, index)
                    del buffer
                    yield output
                    del output
                del values, repetitions
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def _cache_or_compare_frames(self, metric: _RunMetric, table: pd.DataFrame) -> None:
        frames = _validated_frames(table, self.summary)
        if self._frames is None:
            self._frames = frames
        elif not frames.equals(self._frames):
            raise ValueError(
                f"metric {metric.name!r} frame columns disagree with the evaluation run"
            )


def _run_metric(metrics: Mapping[str, _RunMetric], name: str) -> _RunMetric:
    try:
        return metrics[name]
    except KeyError:
        available = ", ".join(repr(key) for key in metrics)
        raise KeyError(f"unknown metric {name!r}; available: {available}") from None


def _read_eager_table(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    columns: list[str] | None = None,
    expand_repetitions: bool = False,
    allow_large: bool,
) -> pd.DataFrame:
    with _open_metric_reader(
        metric,
        summary,
        districts,
        operation,
        expand_repetitions=expand_repetitions,
        allow_large=allow_large,
    ) as parquet:
        return parquet.read(columns=columns, use_threads=False).to_pandas(use_threads=False)


@contextmanager
def _open_metric_reader(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    expand_repetitions: bool = False,
    allow_large: bool,
    batch_size: int | None = None,
) -> Iterator[pq.ParquetFile]:
    with _verified_metric_file(metric) as (file, footer_length):
        estimate = _estimate_memory(
            metric,
            summary,
            districts,
            operation,
            expand_repetitions=expand_repetitions,
            footer_length=footer_length,
            batch_size=batch_size,
        )
        minimum = (
            _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=1,
            )
            if batch_size is not None
            else estimate
        )
        warned = _enforce_memory(
            metric,
            operation,
            estimate,
            allow_large,
            iterator=batch_size is not None,
            can_reduce_batch=minimum < _ERROR_BYTES <= estimate,
        )
        parquet = _validated_parquet_file(file, metric, summary, districts)
        if batch_size is not None:
            metadata = parquet.metadata
            largest_row_group = max(
                (metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)),
                default=0,
            )
            batch_rows = min(summary.accepted, batch_size)
            physical_rows = min(summary.accepted, max(batch_rows, largest_row_group))
            estimate = _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=batch_size,
                physical_rows=physical_rows,
            )
            minimum_rows = min(summary.accepted, max(min(summary.accepted, 1), largest_row_group))
            minimum = _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=1,
                physical_rows=minimum_rows,
            )
            _enforce_memory(
                metric,
                operation,
                estimate,
                allow_large,
                iterator=True,
                can_reduce_batch=minimum < _ERROR_BYTES <= estimate,
                warned=warned,
            )
        yield parquet


@contextmanager
def _verified_metric_file(metric: _RunMetric) -> Iterator[tuple[BinaryIO, int]]:
    with metric.table.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        if size != metric.table_size:
            raise ValueError(f"metric {metric.name!r} table failed its integrity check")
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != metric.table_sha256:
            raise ValueError(f"metric {metric.name!r} table failed its integrity check")
        if size < 12:
            raise ValueError(f"metric {metric.name!r} has an invalid Parquet footer")
        file.seek(-8, os.SEEK_END)
        tail = file.read(8)
        if len(tail) != 8 or tail[4:] != b"PAR1":
            raise ValueError(f"metric {metric.name!r} has invalid Parquet footer magic")
        footer_length = int.from_bytes(tail[:4], "little")
        if footer_length > size - 12:
            raise ValueError(f"metric {metric.name!r} has an invalid Parquet footer length")
        file.seek(0)
        yield file, footer_length


def _validated_parquet_file(
    file: BinaryIO,
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
) -> pq.ParquetFile:
    parquet = pq.ParquetFile(file)
    if parquet.metadata.num_rows != summary.accepted:
        raise ValueError(f"metric {metric.name!r} Parquet row count disagrees with its manifest")
    schema = parquet.schema_arrow
    expected_count = len(_PREFIX_COLUMNS) + _value_column_count(metric, districts)
    if len(schema) != expected_count:
        raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
    prefix_types = (pa.uint64(), pa.uint16(), pa.uint64())
    for index, (name, dtype) in enumerate(zip(_PREFIX_COLUMNS, prefix_types, strict=True)):
        field = schema.field(index)
        if field.name != name:
            raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
        if field.type != dtype:
            raise ValueError(f"metric {metric.name!r} Parquet physical dtypes are unsupported")
    for index, name in enumerate(_value_column_names(metric, districts), len(_PREFIX_COLUMNS)):
        field = schema.field(index)
        if field.name != name:
            raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
        if field.type != pa.float64():
            raise ValueError(f"metric {metric.name!r} Parquet physical dtypes are unsupported")
    return parquet


def _value_column_names(
    metric: _RunMetric,
    districts: tuple[int, ...],
) -> Iterator[str]:
    if metric.shape == "plan":
        yield from metric.subkeys
        return
    for subkey in metric.subkeys:
        for district in districts:
            yield f"{subkey}__district_{district}"


def _value_columns(metric: _RunMetric, districts: tuple[int, ...]) -> list[str]:
    return list(_value_column_names(metric, districts))


def _value_column_count(metric: _RunMetric, districts: tuple[int, ...]) -> int:
    return len(metric.subkeys) if metric.shape == "plan" else len(metric.subkeys) * len(districts)


def _estimate_memory(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    expand_repetitions: bool = False,
    footer_length: int = 0,
    batch_size: int | None = None,
    physical_rows: int | None = None,
) -> int:
    iterator = batch_size is not None
    accepted_rows = summary.accepted if batch_size is None else min(summary.accepted, batch_size)
    if physical_rows is None:
        physical_rows = accepted_rows
    logical_rows = (
        summary.samples
        if batch_size is None and expand_repetitions
        else min(summary.samples, batch_size)
        if batch_size is not None and expand_repetitions
        else accepted_rows
    )
    value_columns = _value_column_count(metric, districts)
    physical = physical_rows * (18 + 8 * value_columns)
    prefix_physical = physical_rows * 18
    accepted_float = accepted_rows * 8 * value_columns
    logical_float = logical_rows * 8 * value_columns
    logical_final = logical_rows * _logical_width(metric, districts)
    frame_cache = accepted_rows * 16
    validation = accepted_rows * 9
    footer = _FOOTER_EXPANSION * footer_length

    if operation == "read":
        index = _semantic_index_bytes(metric, districts, logical_rows)
        total = 2 * physical + accepted_float + logical_float + logical_final + validation + index
        if iterator:
            total += logical_final + index
        else:
            total += frame_cache
    elif operation == "raw":
        index = 16 * (len(_PREFIX_COLUMNS) + value_columns)
        total = 2 * physical + validation + index
        if iterator:
            total += accepted_rows * (18 + 8 * value_columns) + index
        else:
            total += frame_cache
    else:
        index = 32
        total = 2 * prefix_physical + validation + index
        total += 2 * frame_cache
        if iterator:
            total += index
    return (5 * (total + footer) + 3) // 4


def _logical_width(metric: _RunMetric, districts: tuple[int, ...]) -> int:
    multiplier = 1
    if metric.shape == "district":
        multiplier = len(districts)
    elif metric.shape == "region":
        multiplier = len(districts) * len(metric.regions)
    return multiplier * sum(_DTYPE_WIDTHS[dtype] for dtype in metric.dtypes)


def _semantic_index_bytes(
    metric: _RunMetric,
    districts: tuple[int, ...],
    rows: int,
) -> int:
    columns = (
        len(metric.columns) if metric.shape == "plan" else len(metric.columns) * len(districts)
    )
    if metric.shape == "region":
        return 16 * (rows * len(metric.regions) + columns)
    return 16 * columns


def _enforce_memory(
    metric: _RunMetric,
    operation: _ReadOperation,
    estimated_bytes: int,
    allow_large: bool,
    *,
    iterator: bool = False,
    can_reduce_batch: bool = False,
    warned: bool = False,
) -> bool:
    if estimated_bytes < _WARNING_BYTES:
        return warned
    if iterator:
        method = {
            "frames": "iter_frame_batches",
            "raw": "iter_raw_batches",
            "read": "iter_batches",
        }[operation]
        label = f"{method}()" if operation == "frames" else f"{method}({metric.name!r})"
    else:
        label = "frames" if operation == "frames" else f"{operation}({metric.name!r})"
    estimate = _format_bytes(estimated_bytes)
    if not warned:
        _warn_external(f"EvaluationRun.{label} may use approximately {estimate} of memory")
        warned = True
    if estimated_bytes < _ERROR_BYTES or allow_large:
        return warned
    if can_reduce_batch:
        advice = "use a smaller batch_size"
    elif iterator:
        advice = "use allow_large=True"
    elif operation == "frames":
        advice = "use iter_frame_batches()"
    elif operation == "raw":
        advice = "use iter_raw_batches() or allow_large=True"
    else:
        advice = "use iter_batches() or allow_large=True"
    message = (
        f"EvaluationRun.{label} is estimated to use {estimate}, exceeding the "
        f"{_format_bytes(_ERROR_BYTES)} limit; {advice}"
    )
    raise EvaluationMemoryError(message, estimated_bytes, _ERROR_BYTES)


def _warn_external(message: str) -> None:
    frame = inspect.currentframe()
    stacklevel = 1
    while frame is not None and frame.f_globals.get("__name__") in {__name__, "contextlib"}:
        stacklevel += 1
        frame = frame.f_back
    del frame
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def _physical_values(table: pd.DataFrame) -> NDArray[np.float64]:
    return table.iloc[:, len(_PREFIX_COLUMNS) :].to_numpy(dtype=np.float64, copy=True)


def _semantic_value(
    name: str,
    metric: _RunMetric,
    districts: tuple[int, ...],
    flat_values: NDArray[np.float64],
    index: pd.Index,
) -> _EvaluationValue:
    row_count = len(flat_values)
    if metric.shape == "region":
        values = flat_values.reshape(
            row_count,
            len(metric.columns),
            len(metric.regions),
            len(districts),
        )
    elif metric.shape == "district":
        values = flat_values.reshape(row_count, len(metric.columns), len(districts))
    else:
        values = flat_values.reshape(row_count, len(metric.columns))
    result = _MetricResult(
        values,
        metric.shape,
        metric.columns,
        districts if metric.shape != "plan" else (),
        metric.dtypes,
        metric.regions,
        metric.region_name,
    )
    return EnsembleEvalResult({name: result}, _sample_index=index)[name]


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_batch_options(batch_size: object, allow_large: object) -> int:
    if (
        not isinstance(batch_size, numbers.Integral)
        or isinstance(batch_size, (bool, np.bool_))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    _validate_bool(allow_large, "allow_large")
    return int(batch_size)


def _integer_array(series: pd.Series, label: str) -> NDArray[np.integer]:
    values = series.to_numpy(copy=False)
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"evaluation table {label} must contain nonnegative integers")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError(f"evaluation table {label} must contain nonnegative integers")
    return cast("NDArray[np.integer]", values)


def _validated_frame_batch(
    table: pd.DataFrame,
    expected_offset: int,
    expected_accepted: int,
    *,
    return_frame: bool = False,
) -> tuple[pd.DataFrame | None, int, int]:
    if list(table.columns[: len(_PREFIX_COLUMNS)]) != list(_PREFIX_COLUMNS):
        raise ValueError("evaluation table has unsupported prefix columns")
    offsets = _integer_array(cast("pd.Series", table["sample_offset"]), "sample_offset")
    repetitions = _integer_array(cast("pd.Series", table["repetitions"]), "repetitions")
    accepted = _integer_array(cast("pd.Series", table["accepted_index"]), "accepted_index")
    if len(table) == 0:
        frame = (
            pd.DataFrame(
                {
                    "sample_offset": np.array([], dtype=np.int64),
                    "repetitions": np.array([], dtype=np.int64),
                },
                index=pd.RangeIndex(expected_accepted, expected_accepted, name="accepted"),
            )
            if return_frame
            else None
        )
        return frame, expected_offset, expected_accepted
    if int(offsets[0]) != expected_offset:
        raise ValueError("evaluation table sample offsets disagree with repetitions")
    if int(accepted[0]) != expected_accepted:
        raise ValueError("evaluation table accepted indexes are not contiguous")
    if np.any(repetitions == 0):
        raise ValueError("evaluation table repetitions must be positive")
    if len(table) > 1:
        if np.any(offsets[1:] <= offsets[:-1]) or not np.array_equal(
            offsets[1:] - offsets[:-1], repetitions[:-1]
        ):
            raise ValueError("evaluation table sample offsets disagree with repetitions")
        if np.any(accepted[1:] <= accepted[:-1]) or not np.all(accepted[1:] - accepted[:-1] == 1):
            raise ValueError("evaluation table accepted indexes are not contiguous")
    next_offset = int(offsets[-1]) + int(repetitions[-1])
    next_accepted = expected_accepted + len(table)
    frame = None
    if return_frame:
        offset_dtype = np.uint64 if int(offsets[-1]) > np.iinfo(np.int64).max else np.int64
        frame = pd.DataFrame(
            {
                "sample_offset": offsets.astype(offset_dtype, copy=True),
                "repetitions": repetitions.astype(np.int64, copy=True),
            },
            index=pd.RangeIndex(expected_accepted, next_accepted, name="accepted"),
        )
    return frame, next_offset, next_accepted


def _validate_iterator_totals(
    sample_offset: int,
    accepted: int,
    summary: EvaluationSummary,
) -> None:
    if accepted != summary.accepted:
        raise ValueError("evaluation table row count disagrees with the manifest summary")
    if sample_offset != summary.samples:
        raise ValueError("evaluation table sample offsets disagree with repetitions")


def _parse_manifest(
    path: Path,
    manifest: object,
) -> tuple[EvaluationSummary, tuple[int, ...], tuple[_RunMetric, ...]]:
    if not isinstance(manifest, dict):
        raise ValueError("evaluation manifest must be an object")
    data = cast("dict[str, object]", manifest)
    format_version = data.get("format_version")
    if isinstance(format_version, bool) or format_version != 1:
        raise ValueError("evaluation manifest must use format version 1")
    summary_data = data.get("summary")
    if not isinstance(summary_data, dict):
        raise ValueError("evaluation manifest requires a summary")
    summary_data = cast("dict[str, object]", summary_data)
    samples = _nonnegative_int(summary_data.get("samples"), "summary.samples")
    accepted = _nonnegative_int(summary_data.get("accepted"), "summary.accepted")
    if accepted > samples:
        raise ValueError("evaluation manifest accepted frames cannot exceed samples")
    has_unique_plans = "unique_plans" in summary_data
    has_unique_districts = "unique_districts" in summary_data
    if has_unique_plans != has_unique_districts:
        raise ValueError("evaluation manifest unique plan and district counts must appear together")
    unique_plans = (
        _nonnegative_int(summary_data["unique_plans"], "summary.unique_plans")
        if has_unique_plans
        else None
    )
    unique_districts = (
        _nonnegative_int(summary_data["unique_districts"], "summary.unique_districts")
        if has_unique_districts
        else None
    )

    prefix = data.get("prefix_columns")
    expected_prefix = [
        {"name": "sample_offset", "dtype": "uint64"},
        {"name": "repetitions", "dtype": "uint16"},
        {"name": "accepted_index", "dtype": "uint64"},
    ]
    if prefix != expected_prefix:
        raise ValueError("evaluation manifest has unsupported prefix columns")

    district_data = data.get("district_ids")
    if not isinstance(district_data, list):
        raise ValueError("evaluation manifest requires district_ids")
    districts = tuple(_nonnegative_int(value, "district id") for value in district_data)
    if len(set(districts)) != len(districts):
        raise ValueError("evaluation manifest district_ids must be unique")
    if unique_plans is None:
        pass
    elif accepted == 0:
        if unique_plans != 0 or unique_districts != 0:
            raise ValueError("an empty evaluation run cannot contain unique plans or districts")
    else:
        if not 1 <= unique_plans <= accepted:
            raise ValueError(
                "evaluation manifest unique plans must be between one and accepted frames"
            )
        assert unique_districts is not None
        if not len(districts) <= unique_districts <= accepted * len(districts):
            raise ValueError("evaluation manifest unique district count is inconsistent")
    summary = EvaluationSummary(samples, accepted, unique_plans, unique_districts)

    metric_data = data.get("metrics")
    if not isinstance(metric_data, list) or not metric_data:
        raise ValueError("evaluation manifest requires at least one metric")
    metrics = tuple(_parse_run_metric(path, value) for value in metric_data)
    names = [metric.name for metric in metrics]
    if len(set(names)) != len(names):
        raise ValueError("evaluation manifest metric names must be unique")
    return summary, districts, metrics


def _parse_run_metric(path: Path, value: object) -> _RunMetric:
    if not isinstance(value, dict):
        raise ValueError("evaluation manifest metric entries must be objects")
    data = cast("dict[str, object]", value)
    name = data.get("instance")
    if not is_valid_metric_name(name):
        raise ValueError("evaluation manifest contains an invalid metric name")
    assert isinstance(name, str)
    shape_value = data.get("shape")
    if shape_value not in {"district", "plan", "region"}:
        raise ValueError(f"metric {name!r} has an unsupported shape")
    shape = cast("_Shape", shape_value)
    table_name = data.get("table")
    expected_table = f"{name}/scores.parquet"
    if table_name != expected_table:
        raise ValueError(f"metric {name!r} has an unsafe or unsupported table path")
    table = path / expected_table
    if not table.is_file():
        raise FileNotFoundError(f"metric {name!r} table does not exist: {table}")
    table_size = _nonnegative_int(data.get("table_size"), f"metric {name!r} table size")
    table_sha256 = data.get("table_sha256")
    if not isinstance(table_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", table_sha256) is None:
        raise ValueError(f"metric {name!r} requires a valid table SHA-256 digest")

    subkeys = _string_tuple(data.get("subkeys"), f"metric {name!r} subkeys")
    axes = data.get("axes")
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None
    if shape == "region":
        if not isinstance(axes, dict):
            raise ValueError(f"region metric {name!r} requires axis metadata")
        axes = cast("dict[str, object]", axes)
        columns = _string_tuple(axes.get("metric"), f"metric {name!r} axis")
        region = axes.get("region")
        if not isinstance(region, dict):
            raise ValueError(f"region metric {name!r} requires a region axis")
        region = cast("dict[str, object]", region)
        region_name_value = region.get("name")
        if not isinstance(region_name_value, str) or not region_name_value:
            raise ValueError(f"region metric {name!r} requires a region-axis name")
        region_name = region_name_value
        labels = region.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"region metric {name!r} requires region labels")
        regions = tuple(_region_label(label, name) for label in labels)
        if len(set(regions)) != len(regions):
            raise ValueError(f"region metric {name!r} has duplicate region labels")
        expected_subkeys = tuple(
            f"{column}__region_{region_index}"
            for column in columns
            for region_index in range(len(regions))
        )
        if subkeys != expected_subkeys:
            raise ValueError(f"region metric {name!r} subkeys disagree with its axes")
    else:
        if not isinstance(axes, dict):
            raise ValueError(f"metric {name!r} requires axis metadata")
        axes = cast("dict[str, object]", axes)
        if axes.get("region") is not None:
            raise ValueError(f"non-region metric {name!r} cannot define region axes")
        columns = _string_tuple(axes.get("metric"), f"metric {name!r} axis")
        if not subkeys or columns != subkeys:
            raise ValueError(f"metric {name!r} axis values disagree with its subkeys")

    dtypes = _dtype_tuple(data.get("dtypes"), name, len(columns))
    return _RunMetric(
        name,
        shape,
        table,
        subkeys,
        columns,
        dtypes,
        table_size,
        table_sha256,
        regions,
        region_name,
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must contain unique nonempty strings")
    return cast("tuple[str, ...]", tuple(value))


def _dtype_tuple(value: object, name: str, count: int) -> tuple[_Dtype, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(dtype not in _PANDAS_DTYPES for dtype in value)
    ):
        raise ValueError(f"metric {name!r} has invalid logical dtypes")
    return cast("tuple[_Dtype, ...]", tuple(value))


def _region_label(value: object, name: str) -> Hashable:
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        raise ValueError(f"region metric {name!r} has an invalid region label")
    data = cast("dict[str, object]", value)
    kind = data["kind"]
    label = data["value"]
    if kind == "str" and isinstance(label, str):
        return label
    if kind == "int" and isinstance(label, int) and not isinstance(label, bool):
        return label
    raise ValueError(f"region metric {name!r} has an invalid region label")


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)) or value < 0:
        raise ValueError(f"evaluation manifest {label} must be a nonnegative integer")
    return int(value)


def _validated_frames(table: pd.DataFrame, summary: EvaluationSummary) -> pd.DataFrame:
    if len(table) != summary.accepted:
        raise ValueError("evaluation table row count disagrees with the manifest summary")
    frames, sample_offset, accepted = _validated_frame_batch(table, 0, 0, return_frame=True)
    assert frames is not None
    _validate_iterator_totals(sample_offset, accepted, summary)
    return frames
