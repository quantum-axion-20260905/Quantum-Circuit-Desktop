import hashlib
import json

from django.core.exceptions import ValidationError
from django.db import models


class Project(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class CircuitVersion(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="versions")
    created_at = models.DateTimeField(auto_now_add=True)
    # OpenQASM 3 + optional UI metadata JSON
    qasm = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["project", "-created_at"], name="circuit_project_created_idx")]

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            {"qasm": self.qasm, "metadata": self.metadata},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, *args, **kwargs):
        if self.pk is not None:
            raise ValidationError("CircuitVersion is immutable; create a new version instead.")
        if not self.qasm.strip():
            raise ValidationError("qasm must not be empty")
        return super().save(*args, **kwargs)


class Run(models.Model):
    """
    A reproducible compute run tied to a specific circuit version.
    Stores request/response artifacts (JSON) for replay and audit.
    """

    kind = models.CharField(max_length=64)  # e.g. tn_estimate, tn_amplitudes
    version = models.ForeignKey(CircuitVersion, on_delete=models.CASCADE, related_name="runs")
    created_at = models.DateTimeField(auto_now_add=True)

    status = models.CharField(max_length=20, default="running")  # running|done|failed
    seed = models.BigIntegerField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    agent_base_url = models.CharField(max_length=300, blank=True, default="")
    agent_job_id = models.CharField(max_length=64, blank=True, default="")
    backend = models.CharField(max_length=100, blank=True, default="")  # backend name in agent
    device = models.JSONField(default=dict, blank=True)  # GPU/CPU snapshot from agent

    request = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["version", "-created_at"], name="run_version_created_idx"),
            models.Index(fields=["status", "-created_at"], name="run_status_created_idx"),
            models.Index(fields=["kind", "-created_at"], name="run_kind_created_idx"),
        ]


class RunArtifact(models.Model):
    """
    Normalized artifacts for indexing and future external storage.
    """

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="artifacts")
    kind = models.CharField(max_length=64)  # counts|amplitudes|estimate|raw
    created_at = models.DateTimeField(auto_now_add=True)
    content = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["run", "kind", "-created_at"], name="artifact_run_kind_idx")]


class Study(models.Model):
    """Durable manifest for a bounded multi-run research study.

    Individual Run rows remain the source for point-level provenance.  The
    study stores the immutable campaign description and aggregate point
    results so a convergence campaign can be listed and resumed as one unit.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="studies")
    kind = models.CharField(max_length=64)  # e.g. dmrg-convergence
    label = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, default="running")  # running|done|failed|canceled
    error = models.TextField(blank=True, default="")
    request = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["project", "-created_at"], name="study_project_created_idx"),
            models.Index(fields=["status", "-created_at"], name="study_status_created_idx"),
            models.Index(fields=["kind", "-created_at"], name="study_kind_created_idx"),
        ]
