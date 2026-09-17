from rest_framework import viewsets
from django.utils import timezone

from .models import CircuitVersion, Project, Run, RunArtifact, Study
from .serializers import CircuitVersionSerializer, ProjectSerializer, RunArtifactSerializer, RunSerializer, StudySerializer


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.order_by("-created_at")
    serializer_class = ProjectSerializer


class CircuitVersionViewSet(viewsets.ModelViewSet):
    queryset = CircuitVersion.objects.order_by("-created_at")
    serializer_class = CircuitVersionSerializer
    http_method_names = ["get", "post", "head", "options"]

    def get_queryset(self):
        qs = super().get_queryset()
        project = self.request.query_params.get("project")
        if project:
            qs = qs.filter(project_id=project)
        return qs


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
        # Reads stay read-only and use the relationship/indexes introduced for
        # long run histories. Legacy reconciliation is opt-in below so a list
        # request cannot unexpectedly write every stale row.
        qs = Run.objects.select_related("version", "version__project").order_by("-created_at")
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

    def list(self, request, *args, **kwargs):
        if request.query_params.get("reconcile") == "1":
            self._reconcile_stale_runs()
        return super().list(request, *args, **kwargs)


class RunArtifactViewSet(viewsets.ModelViewSet):
    queryset = RunArtifact.objects.all().order_by("-created_at")
    serializer_class = RunArtifactSerializer

    def get_queryset(self):
        qs = super().get_queryset().select_related("run")
        qp = self.request.query_params
        run = qp.get("run")
        if run:
            qs = qs.filter(run_id=run)
        kind = qp.get("kind")
        if kind:
            qs = qs.filter(kind=kind)
        return qs


class StudyViewSet(viewsets.ModelViewSet):
    queryset = Study.objects.select_related("project").order_by("-created_at")
    serializer_class = StudySerializer
    http_method_names = ["get", "post", "head", "options"]

    def get_queryset(self):
        qs = super().get_queryset()
        project = self.request.query_params.get("project")
        if project:
            qs = qs.filter(project_id=project)
        kind = self.request.query_params.get("kind")
        if kind:
            qs = qs.filter(kind=kind)
        status = self.request.query_params.get("status")
        if status:
            qs = qs.filter(status=status)
        return qs
