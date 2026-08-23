from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from Agent.cli import run_train_bridge


class AgentBridgeTests(unittest.TestCase):
    @patch("Agent.cli.subprocess.run")
    def test_documented_separator_is_not_forwarded_to_child_parser(self, mocked_run) -> None:
        mocked_run.return_value.returncode = 0
        args = argparse.Namespace(
            command="prepare-scene",
            forwarded=["--", "--dataset", "local-data", "--output", "artifacts"],
        )

        with self.assertRaises(SystemExit) as raised:
            run_train_bridge(args)

        self.assertEqual(raised.exception.code, 0)
        command = mocked_run.call_args.args[0]
        self.assertEqual(command[-4:], ["--dataset", "local-data", "--output", "artifacts"])
        self.assertNotIn("--", command)


if __name__ == "__main__":
    unittest.main()
