from run import build_parser, pipeline_steps


def steps(*argv):
    return pipeline_steps(build_parser().parse_args(list(argv)))


def test_daily_batch_steps():
    s = steps("/c", "--out", "/o", "--list", "/b.txt", "--device", "mps", "--no-report")
    assert [x[0] for x in s] == ["scan.py", "places.py", "lurkers.py", "bikes.py"]
    assert s[0][s[0].index("--list") + 1] == "/b.txt" and s[0][s[0].index("--model") + 1] == "rfdetr:large"
    assert s[2][s[2].index("--stage") + 1] == "tracks" and s[3][s[3].index("--stage") + 1] == "scan"
    assert all(x[x.index("--root") + 1] == "/c" for x in s[1:])
    assert all(x[x.index("--device") + 1] == "mps" for x in s)


def test_report_runs_last_unless_disabled():
    assert steps("/c", "--out", "/o")[-1][0] == "report.py"


def test_shard_goes_to_scan_only():
    s = steps("/c", "--out", "/o", "--shard", "1/3", "--no-report")
    assert "--shard" in s[0] and not any("--shard" in x for x in s[1:])
