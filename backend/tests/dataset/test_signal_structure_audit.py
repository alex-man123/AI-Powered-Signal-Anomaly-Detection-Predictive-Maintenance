from app.datasets.signal_structure_audit import inspect_file, select_audit_files


def test_selects_at_least_three_files_across_distinct_states() -> None:
    selection = select_audit_files()

    assert len(selection) >= 3

    states = {state for _relative_path, state in selection}
    assert len(states) == len(selection)  # one file per distinct state, no duplicates


def test_each_selected_file_has_rows_and_identified_columns() -> None:
    selection = select_audit_files()

    for relative_path, state in selection:
        structure = inspect_file(relative_path, state)

        assert structure.rows > 0
        assert structure.columns > 0
        assert len(structure.column_dtypes) == structure.columns


def test_column_count_and_dtype_are_consistent_across_all_selected_states() -> None:
    """Empirically confirmed by this audit: every state's representative file has the
    same number of float64 columns and no missing values. These are measured facts
    from the local dataset, not values copied from MAFAULDA documentation."""
    selection = select_audit_files()
    structures = [inspect_file(relative_path, state) for relative_path, state in selection]

    column_counts = {structure.columns for structure in structures}
    dtype_sets = {structure.column_dtypes for structure in structures}

    assert column_counts == {8}
    assert dtype_sets == {("float64",) * 8}
    assert all(not structure.has_missing_values for structure in structures)
