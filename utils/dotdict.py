"""DotDict: a dict with attribute access, for the YAML configs.

Replaces the `dotwiz` package, whose PyPI build pulls in a Rust-compiled dependency
(pyheck) that cannot be built on Rorqual login nodes.

Nested dicts, including dicts inside lists, become DotDicts on construction, on item or
attribute assignment and on update(). to_dict() converts back to plain dicts and lists,
e.g. for yaml.safe_dump or json.dump.
"""


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, DotDict):
        return DotDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value):
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


class DotDict(dict):
    """dict subclass where d.key is d['key']; dict methods win over keys of the same name."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __getattr__(self, name):
        # Only called when normal lookup fails, so dict methods and dunders are unaffected.
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __reduce__(self):
        # Pickle (DataLoader workers, deepcopy) through the plain-dict form.
        return (DotDict, (self.to_dict(),))

    def copy(self):
        return DotDict(self)

    def to_dict(self):
        """Plain dicts and lists all the way down."""
        return _unwrap(self)

    def __repr__(self):
        return f'DotDict({dict.__repr__(self)})'


def config_dict(params):
    """A plain dict of a DotDict/dict config or an argparse Namespace, for dumping to YAML."""
    if isinstance(params, dict):
        return _unwrap(params)
    return _unwrap(vars(params))
