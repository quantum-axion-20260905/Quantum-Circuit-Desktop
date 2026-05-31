from rest_framework import viewsets
from django.utils import timezone

from .models import CircuitVersion, Project, Run, RunArtifact
from .serializers import CircuitVersionSerializer, ProjectSerializer, RunArtifactSerializer, RunSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.order_by("-created_at")
    serializer_class = ProjectSerializer


class CircuitVersionViewSet(viewsets.ModelViewSet):
    queryset = CircuitVersion.objects.order_by("-created_at")
    serializer_class = CircuitVersionSerializer


class RunViewSet(viewsets.ModelViewSet):
    queryset = Run.objects.all()
    serializer_class = RunSerializer

    @staticmethod
    def _reconcile_stale_runs() -> None:
        """
        Normalize historical rows where result/error exists but status stayed "running".
        This keeps UI and API status consistent after older client/agent flows.
        """
        stale = Run.objects.filter(status="running")
        for run in stale.iterator():
            has_result = isinstance(run.result, dict) and bool(run.result)
            has_error = bool(run.error)
            if not has_result and not has_error:
                continue
            run.status = "failed" if has_error else "done"
            if run.finished_at is None:
                run.finished_at = timezone.now()
            run.save(update_fields=["status", "finished_at"])

    def get_queryset(self):
        self._reconcile_stale_runs()
        qs = Run.objects.order_by("-created_at")
        qp = self.request.query_params

        version = qp.get("version")
        if version:
            qs = qs.filter(version_id=version)

        project = qp.get("project")
        if project:
            qs = qs.filter(version__project_id=project)

        kind = qp.get("kind")
        if kind:
            qs = qs.filter(kind=kind)

        status = qp.get("status")
        if status:
            qs = qs.filter(status=status)

        return qs


class RunArtifactViewSet(viewsets.ModelViewSet):
    queryset = RunArtifact.objects.all().order_by("-created_at")
    serializer_class = RunArtifactSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        qp = self.request.query_params
        run = qp.get("run")
        if run:
            qs = qs.filter(run_id=run)
        kind = qp.get("kind")
        if kind:
            qs = qs.filter(kind=kind)
        return qs
