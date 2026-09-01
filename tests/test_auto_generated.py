"""Auto-generated smoke tests. Regenerate with: python -m portautomation.testing.generator"""

import importlib

import pytest

MODULES = ['portautomation.config', 'portautomation.data', 'portautomation.dataset', 'portautomation.device', 'portautomation.evaluate', 'portautomation.gpu_env', 'portautomation.metrics', 'portautomation.models', 'portautomation.pipeline', 'portautomation.train', 'portautomation.validation', 'portautomation.visualize']

def test_import_portautomation_config():
    module = importlib.import_module('portautomation.config')
    assert module is not None

def test_import_portautomation_data():
    module = importlib.import_module('portautomation.data')
    assert module is not None

def test_import_portautomation_dataset():
    module = importlib.import_module('portautomation.dataset')
    assert module is not None

def test_import_portautomation_device():
    module = importlib.import_module('portautomation.device')
    assert module is not None

def test_import_portautomation_evaluate():
    module = importlib.import_module('portautomation.evaluate')
    assert module is not None

def test_import_portautomation_gpu_env():
    module = importlib.import_module('portautomation.gpu_env')
    assert module is not None

def test_import_portautomation_metrics():
    module = importlib.import_module('portautomation.metrics')
    assert module is not None

def test_import_portautomation_models():
    module = importlib.import_module('portautomation.models')
    assert module is not None

def test_import_portautomation_pipeline():
    module = importlib.import_module('portautomation.pipeline')
    assert module is not None

def test_import_portautomation_train():
    module = importlib.import_module('portautomation.train')
    assert module is not None

def test_import_portautomation_validation():
    module = importlib.import_module('portautomation.validation')
    assert module is not None

def test_import_portautomation_visualize():
    module = importlib.import_module('portautomation.visualize')
    assert module is not None

@pytest.mark.parametrize('module_name', MODULES)
def test_module_has_docstring(module_name):
    module = importlib.import_module(module_name)
    assert module.__doc__ is not None

def test_public_functions_exist_portautomation_data():
    module = importlib.import_module('portautomation.data')
    expected = ['build_split_datasets', 'compute_class_weights', 'load_images_to_dataframe', 'save_images_to_directory', 'seed_everything', 'split_dataframe', 'summarize_labels', 'validate_class_names', 'validate_directory', 'validate_image_dataframe', 'validate_image_scaling', 'validate_image_size', 'validate_path_writable', 'validate_positive_int', 'validate_ratio', 'warn_class_imbalance', 'warn_small_split']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_dataset():
    module = importlib.import_module('portautomation.dataset')
    expected = ['ensure_dataset', 'validate_directory']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_device():
    module = importlib.import_module('portautomation.device')
    expected = ['configure_devices', 'device_dict', 'get_device_info', 'optimize_dataset']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_evaluate():
    module = importlib.import_module('portautomation.evaluate')
    expected = ['build_confusion_matrix', 'collect_predictions', 'evaluate_model', 'print_classification_report', 'validate_class_names', 'validate_confusion_matrix', 'validate_label_array']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_gpu_env():
    module = importlib.import_module('portautomation.gpu_env')
    expected = ['discover_nvidia_lib_dirs', 'ensure_gpu_environment', 'preload_nvidia_libraries', 'setup_nvidia_library_path']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_metrics():
    module = importlib.import_module('portautomation.metrics')
    expected = ['precision_m', 'recall_m']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_models():
    module = importlib.import_module('portautomation.models')
    expected = ['build_cnn', 'build_mobilenet', 'compile_classifier', 'precision_m', 'recall_m', 'validate_image_size', 'validate_positive_int']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_pipeline():
    module = importlib.import_module('portautomation.pipeline')
    expected = ['build_cnn', 'build_confusion_matrix', 'build_mobilenet', 'build_split_datasets', 'collect_predictions', 'compile_classifier', 'compute_class_weights', 'ensure_dataset', 'evaluate_model', 'load_images_to_dataframe', 'main', 'parse_args', 'plot_confusion_matrix', 'plot_training_history', 'print_classification_report', 'run_cnn', 'run_mobilenet', 'save_model', 'seed_everything', 'split_dataframe', 'summarize_labels', 'train_cnn', 'train_mobilenet']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_train():
    module = importlib.import_module('portautomation.train')
    expected = ['save_model', 'train_cnn', 'train_mobilenet', 'validate_non_negative_int', 'validate_path', 'validate_positive_int']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_validation():
    module = importlib.import_module('portautomation.validation')
    expected = ['validate_class_names', 'validate_confusion_matrix', 'validate_directory', 'validate_image_dataframe', 'validate_image_scaling', 'validate_image_size', 'validate_label_array', 'validate_non_negative_int', 'validate_path', 'validate_positive_int', 'validate_ratio', 'warn_class_imbalance', 'warn_small_split']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_public_functions_exist_portautomation_visualize():
    module = importlib.import_module('portautomation.visualize')
    expected = ['plot_confusion_matrix', 'plot_training_history', 'validate_class_names', 'validate_confusion_matrix', 'validate_path']
    for name in expected:
        assert hasattr(module, name), f'Missing function: {name}'

def test_config_paths_exist():
    from portautomation import config
    assert config.PROJECT_ROOT.exists()
    assert config.NUM_CLASSES == 9
    assert config.IMAGE_SIZE == (224, 224)

def test_class_names_count_matches_num_classes():
    from portautomation import config
    assert len(config.CLASS_NAMES) == config.NUM_CLASSES
