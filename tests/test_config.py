from portautomation import config


def test_project_root_exists():
    assert config.PROJECT_ROOT.exists()


def test_image_size_is_square():
    assert config.IMAGE_SIZE[0] == config.IMAGE_SIZE[1]


def test_batch_size_positive():
    assert config.BATCH_SIZE > 0


def test_data_zip_parts_sorted():
    assert config.DATA_ZIP_PARTS == sorted(config.DATA_ZIP_PARTS)
