from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("projects", "0004_scaling_indexes")]

    operations = [
        migrations.CreateModel(
            name="Study",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(max_length=64)),
                ("label", models.CharField(max_length=200)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("status", models.CharField(default="running", max_length=20)),
                ("error", models.TextField(blank=True, default="")),
                ("request", models.JSONField(blank=True, default=dict)),
                ("result", models.JSONField(blank=True, default=dict)),
                ("project", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="studies", to="projects.project")),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddIndex(
            model_name="study",
            index=models.Index(fields=["project", "-created_at"], name="study_project_created_idx"),
        ),
        migrations.AddIndex(
            model_name="study",
            index=models.Index(fields=["status", "-created_at"], name="study_status_created_idx"),
        ),
        migrations.AddIndex(
            model_name="study",
            index=models.Index(fields=["kind", "-created_at"], name="study_kind_created_idx"),
        ),
    ]
