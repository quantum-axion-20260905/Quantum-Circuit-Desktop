from rest_framework import serializers

from .models import CircuitVersion, Project, Run, RunArtifact


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ["id", "name", "created_at"]


class CircuitVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CircuitVersion
        fields = ["id", "project", "created_at", "qasm", "metadata"]


class RunSerializer(serializers.ModelSerializer):
    class Meta:
        model = Run
        fields = [
            "id",
            "kind",
            "version",
            "created_at",
            "status",
            "seed",
            "started_at",
            "finished_at",
            "error",
            "agent_base_url",
            "agent_job_id",
            "backend",
            "device",
            "request",
            "result",
        ]


class RunArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = RunArtifact
        fields = ["id", "run", "kind", "created_at", "content"]
