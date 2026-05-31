from django.db import models


class Project(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)


class CircuitVersion(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="versions")
    created_at = models.DateTimeField(auto_now_add=True)
    # OpenQASM 3 + optional UI metadata JSON
    qasm = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)


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


class RunArtifact(models.Model):
    """
    Normalized artifacts for indexing and future external storage.
    """

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="artifacts")
    kind = models.CharField(max_length=64)  # counts|amplitudes|estimate|raw
    created_at = models.DateTimeField(auto_now_add=True)
    content = models.JSONField(default=dict, blank=True)
