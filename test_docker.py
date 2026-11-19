import subprocess
import unittest

from minimal_agent.environments.docker import DockerEnvironment
from minimal_agent.exceptions import Submitted, TimeExceeded


class DockerEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = DockerEnvironment(image="python:3.12-slim", cwd="/tmp", env={"SAMPLE": "test-value"})
        self.addCleanup(self.env.close)

    def assert_container_removed(self, container_id):
        result = subprocess.run(
            [self.env.config.executable, "container", "ls", "-aq", "--filter", f"id={container_id}"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "")

    def test_execute_and_submit(self):
        write = self.env.execute({"command": 'pwd && printf %s "$SAMPLE" > marker'})
        self.assertEqual((write["returncode"], write["output"]), (0, "/tmp\n"))
        self.assertEqual(self.env.execute({"command": "cat marker"})["output"], "test-value")
        failure = self.env.execute({"command": "echo error >&2; exit 7"})
        self.assertEqual((failure["returncode"], failure["output"]), (7, "error\n"))
        with self.assertRaises(Submitted) as submitted:
            self.env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo patch"})
        self.assertEqual(submitted.exception.messages[0]["extra"]["submission"], "patch\n")
        container_id = self.env.container_id
        self.env.close()
        self.env.close()
        self.assert_container_removed(container_id)

    def test_timeout_ends_instance(self):
        container_id = self.env.container_id
        with self.assertRaises(TimeExceeded) as expired:
            self.env.execute({"command": "echo started; sleep 20"}, timeout=2)
        message = expired.exception.messages[0]
        self.assertEqual(message["extra"]["output"], "started\n")
        self.assertNotIn("test-value", str(message))
        self.assert_container_removed(container_id)


if __name__ == "__main__":
    unittest.main()

