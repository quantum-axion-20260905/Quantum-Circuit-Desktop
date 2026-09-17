from django.core.exceptions import ValidationError
from django.test import TestCase

from .models import CircuitVersion, Project, Study


class CircuitVersionTests(TestCase):
    def test_fingerprint_is_stable_for_same_content(self):
        project = Project.objects.create(name="test")
        first = CircuitVersion.objects.create(project=project, qasm="OPENQASM 3;", metadata={"ui": {}})
        second = CircuitVersion.objects.create(project=project, qasm="OPENQASM 3;", metadata={"ui": {}})
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)

    def test_existing_version_cannot_be_overwritten(self):
        project = Project.objects.create(name="test")
        version = CircuitVersion.objects.create(project=project, qasm="OPENQASM 3;")
        version.qasm = "OPENQASM 3;\nqubit[2] q;"
        with self.assertRaises(ValidationError):
            version.save()


class StudyTests(TestCase):
    def test_study_manifest_keeps_campaign_summary(self):
        project = Project.objects.create(name="study")
        study = Study.objects.create(
            project=project,
            kind="dmrg-convergence",
            label="DMRG study",
            status="done",
            request={"mode": "dmrg", "points": [{"label": "chi=2"}]},
            result={"completed": 1, "points": [{"status": "done", "energy": -1.0}]},
        )
        self.assertEqual(project.studies.get(pk=study.pk).result["completed"], 1)
        self.assertEqual(study.kind, "dmrg-convergence")
