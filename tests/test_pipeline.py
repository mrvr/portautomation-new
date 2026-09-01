from portautomation.pipeline import parse_args, run_cnn, run_mobilenet


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    args = parse_args()
    assert args.model == "both"


def test_run_cnn_and_mobilenet_are_callable():
    assert callable(run_cnn)
    assert callable(run_mobilenet)
