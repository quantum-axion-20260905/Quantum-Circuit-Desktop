from django.contrib import admin

from .models import CircuitVersion, Project, Run, RunArtifact, Study

admin.site.register(Project)
admin.site.register(CircuitVersion)
admin.site.register(Run)
admin.site.register(RunArtifact)
admin.site.register(Study)
