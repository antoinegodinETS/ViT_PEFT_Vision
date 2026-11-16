"""utils.dotdict.DotDict, the local replacement for the dotwiz package."""
import argparse
import copy
import json
import pickle

import pytest
import yaml

from conftest import REPO
from utils.dotdict import DotDict, config_dict
from utils.misc import get_param, load_yaml


def sample():
    return DotDict({'lr': 0.1, 'model': {'name': 'vit', 'depth': 12},
                    'stages': [{'k': 1}, 2, [{'deep': True}]]})


def test_nested_dicts_and_lists_are_converted():
    d = sample()
    assert isinstance(d.model, DotDict)
    assert d.model.name == 'vit'
    assert isinstance(d.stages[0], DotDict) and d.stages[0].k == 1
    assert d.stages[1] == 2
    assert d.stages[2][0].deep is True


def test_attribute_and_item_access():
    d = sample()
    assert d.lr == d['lr'] == 0.1
    d.epoch = 5
    assert d['epoch'] == 5
    d['batch_size'] = 64
    assert d.batch_size == 64
    d.extra = {'a': {'b': 1}}
    assert d.extra.a.b == 1
    del d.epoch
    assert 'epoch' not in d
    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(AttributeError):
        del d.missing
    with pytest.raises(KeyError):
        d['missing']


def test_dict_protocol():
    d = sample()
    assert isinstance(d, dict)
    assert 'lr' in d and 'nope' not in d
    assert d.get('lr') == 0.1 and d.get('nope', 3) == 3
    assert list(d) == ['lr', 'model', 'stages']
    assert [k for k, _ in d.items()] == ['lr', 'model', 'stages']
    assert d.pop('lr') == 0.1 and 'lr' not in d
    assert d.pop('lr', None) is None
    assert d.setdefault('x', {'y': 1}).y == 1


def test_dict_methods_win_over_keys():
    d = DotDict({'items': 3, 'update': 4})
    assert d['items'] == 3
    assert sorted(k for k, _ in d.items()) == ['items', 'update']


def test_update_converts_and_overrides():
    d = sample()
    d.update({'lr': 0.01, 'opt': {'name': 'adam'}}, wd=1e-4)
    assert d.lr == 0.01 and d.wd == 1e-4
    assert isinstance(d.opt, DotDict) and d.opt.name == 'adam'


def test_to_dict_is_plain_and_dumps():
    d = sample()
    plain = d.to_dict()
    assert type(plain) is dict and type(plain['model']) is dict
    assert type(plain['stages'][0]) is dict and type(plain['stages'][2][0]) is dict
    assert plain == {'lr': 0.1, 'model': {'name': 'vit', 'depth': 12},
                     'stages': [{'k': 1}, 2, [{'deep': True}]]}
    assert yaml.safe_load(yaml.safe_dump(plain)) == plain
    assert json.loads(json.dumps(d)) == plain


def test_config_dict_accepts_dotdict_dict_and_namespace():
    assert config_dict(sample()) == sample().to_dict()
    assert config_dict({'a': DotDict({'b': 1})}) == {'a': {'b': 1}}
    ns = argparse.Namespace(data='x', lr=0.1)
    assert config_dict(ns) == {'data': 'x', 'lr': 0.1}
    yaml.safe_dump(config_dict(sample()))


def test_copy_pickle_deepcopy():
    d = sample()
    shallow = d.copy()
    assert isinstance(shallow, DotDict) and shallow == d
    shallow.lr = 9
    assert d.lr == 0.1
    for clone in (pickle.loads(pickle.dumps(d)), copy.deepcopy(d)):
        assert isinstance(clone, DotDict) and clone == d
        assert isinstance(clone.model, DotDict)
        clone.model.depth = 1
        assert d.model.depth == 12


def test_get_param_reads_dotdict():
    d = DotDict({'lora_alpha': 16})
    assert get_param(d, 'lora_alpha') == 16
    assert get_param(d, 'lora_dropout', 0.1) == 0.1


def test_load_yaml_returns_dotdict(tmp_path):
    path = tmp_path / 'c.yml'
    path.write_text('lr: [0.1, 0.01]\nper_task:\n  vpt_num:\n    cifar: 10\n')
    cfg = load_yaml(str(path))
    assert isinstance(cfg, DotDict)
    assert cfg.lr == [0.1, 0.01]
    assert cfg.per_task.vpt_num.cifar == 10


def test_load_yaml_real_methods_file():
    methods = load_yaml(f'{REPO}/experiment/config/sys843/methods.yml')
    assert isinstance(methods, DotDict) and len(methods) > 0
    for entry in methods.values():
        assert isinstance(entry, DotDict) and 'tune' in entry
