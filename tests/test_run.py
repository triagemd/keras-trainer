import pytest
import os
import json
import base64
import shutil
import subprocess

from stored import list_files


@pytest.fixture('function')
def output_dir():
    path = '.cache/catdog/output'
    if os.path.exists(path):
        shutil.rmtree(path)
    return path


def test_run_with_model_spec_name_only(output_dir):
    model_output_dir = os.path.join(output_dir, 'model')
    logs_output_dir = os.path.join(output_dir, 'logs')
    subprocess.check_output([
        'python',
        '-m',
        'keras_trainer.run',
        '--model_spec',
        'mobilenet_v1',
        '--num_classes',
        '2',
        '--train_dataset_dir',
        os.path.abspath('tests/files/catdog/train'),
        '--val_dataset_dir',
        os.path.abspath('tests/files/catdog/val'),
        '--output_model_dir',
        model_output_dir,
        '--output_logs_dir',
        logs_output_dir
    ])

    actual = list_files(model_output_dir, relative=True)
    assert len(actual) == 5

    actual = list_files(logs_output_dir, relative=True)
    assert len(actual) > 0
    for path in actual:
        assert path.startswith('events.out.tfevents.'), path


def test_run_with_model_spec_encoded_base64(output_dir):
    model_output_dir = os.path.join(output_dir, 'model')
    logs_output_dir = os.path.join(output_dir, 'logs')
    subprocess.check_output([
        'python',
        '-m',
        'keras_trainer.run',
        '--model_spec',
        base64.b64encode(json.dumps({'name': 'mobilenet_v1'}).encode()),
        '--num_classes',
        '2',
        '--train_dataset_dir',
        'tests/files/catdog/train',
        '--val_dataset_dir',
        'tests/files/catdog/val',
        '--output_model_dir',
        model_output_dir,
        '--output_logs_dir',
        logs_output_dir
    ])

    actual = list_files(model_output_dir, relative=True)
    assert len(actual) == 5

    actual = list_files(logs_output_dir, relative=True)
    assert len(actual) > 0
    for path in actual:
        assert path.startswith('events.out.tfevents.'), path
