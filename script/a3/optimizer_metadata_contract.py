"""Compare Adam metadata across Python checkpoints and JSON receipts."""

import math


def canonical_groups(groups):
    if type(groups) is not list:
        raise ValueError('Optimizer groups must be a list')
    result = []
    for group in groups:
        if type(group) is not dict or 'betas' not in group:
            raise ValueError('Missing optimizer group/betas')
        betas = group['betas']
        if (type(betas) not in (tuple, list) or len(betas) != 2 or
                any(type(v) is not float or not math.isfinite(v) or not 0 <= v < 1 for v in betas)):
            raise ValueError('Invalid Adam betas')
        row = {}
        for key, value in group.items():
            if type(key) is not str:
                raise ValueError('Invalid metadata key')
            if key == 'betas':
                row[key] = ('betas', tuple(betas))
            elif type(value) in (str, bool, int, float) or value is None:
                if type(value) is float and not math.isfinite(value):
                    raise ValueError('Nonfinite metadata')
                # Preserve exact types: True must not silently match numeric 1.
                row[key] = (type(value).__name__, value)
            else:
                raise ValueError('Unsupported metadata field: ' + key)
        result.append(row)
    return result


def require_optimizer_receipt(mapping, groups, receipt):
    if mapping != receipt['parameters']:
        raise ValueError('Original optimizer parameter mapping changed')
    if canonical_groups(groups) != canonical_groups(receipt['groups']):
        raise ValueError('Original optimizer flags or scalar values changed')
    return dict(parameters_exact=True, metadata_exact=True,
                only_representation_conversion='BETAS_TUPLE_OR_JSON_LIST')
