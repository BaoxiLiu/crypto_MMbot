"""Compose explicit JSON profiles; resolved contents and hashes travel with a run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SECTIONS={
    'data': {'window','symbols','datasets','rules'},
    'execution': {'engine','execution','costs'},
    'risk': {'account','loss_stops'},
}


def merge_parameters(base, overrides):
    """Explicit recursive parameter merge; scalars/lists replace in full."""
    if not isinstance(base,dict) or not isinstance(overrides,dict):
        raise ValueError('parameter_overrides must be an object')
    result=deepcopy(base)
    for key,value in overrides.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=merge_parameters(result[key],value)
        else:
            result[key]=deepcopy(value)
    return result


def expand_profiles(original, root=ROOT):
    spec=deepcopy(original)
    inputs={}

    def read(value):
        p=Path(value);p=p if p.is_absolute() else Path(root)/p
        p=p.resolve();raw=p.read_bytes()
        inputs[str(p)]=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if not isinstance(data,dict):raise ValueError('Profile must be a JSON object: '+str(p))
        return data

    profiles=spec.pop('profiles',{})
    if not isinstance(profiles,dict) or set(profiles)-set(SECTIONS):
        raise ValueError('Unknown experiment profile section')
    for kind,path in profiles.items():
        profile=read(path)
        if set(profile)-SECTIONS[kind]:raise ValueError(f'{kind} profile crosses module boundary')
        for key,value in profile.items():
            if key in spec and spec[key]!=value:
                raise ValueError('Conflicting inline/profile value: '+key)
            spec[key]=value
    for case in spec.get('strategies',[]):
        path=case.pop('profile',None)
        has_overrides='parameter_overrides' in case
        parameter_overrides=case.pop('parameter_overrides',None)
        if has_overrides and path is None:
            raise ValueError('parameter_overrides requires a strategy profile')
        if path is not None:
            profile=read(path)
            if set(profile)!={'type','version','params'}:raise ValueError('Strategy profile requires type/version/params')
            if 'plugin' in case:raise ValueError('Choose strategy profile or inline plugin')
            case['plugin']=profile
            if has_overrides:
                profile['params']=merge_parameters(profile['params'],parameter_overrides)
        plugin=case.get('plugin')
        if plugin is not None:
            plugin=deepcopy(plugin);case['plugin']=plugin
            if set(plugin)!={'type','version','params'}:raise ValueError('Plugin requires type/version/params')
            params=plugin['params']
            if not isinstance(params,dict):raise ValueError('Plugin params must be an object')
            if 'model_profile' in params:
                raise ValueError('model_profile is retired; use the selected plugin model parameters')
    if inputs:spec['_profile_inputs']=inputs
    return spec
