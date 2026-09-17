from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("projects", "0003_run_agent_job_id_runartifact")]

    operations = [
        migrations.AlterModelOptions(
            name="project",
            options={"ordering": ["-created_at"]},
        ),
        migrations.AlterModelOptions(
            name="circuitversion",
            options={"ordering": ["-created_at"]},
        ),
        migrations.AlterModelOptions(
            name="run",
            options={"ordering": ["-created_at"]},
        ),
        migrations.AlterModelOptions(
            name="runartifact",
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddIndex(
            model_name="circuitversion",
            index=models.Index(fields=["project", "-created_at"], name="circuit_project_created_idx"),
        ),
        migrations.AddIndex(
            model_name="run",
            index=models.Index(fields=["version", "-created_at"], name="run_version_created_idx"),
        ),
        migrations.AddIndex(
            model_name="run",
            index=models.Index(fields=["status", "-created_at"], name="run_status_created_idx"),
        ),
        migrations.AddIndex(
            model_name="run",
            index=models.Index(fields=["kind", "-created_at"], name="run_kind_created_idx"),
        ),
        migrations.AddIndex(
            model_name="runartifact",
            index=models.Index(fields=["run", "kind", "-created_at"], name="artifact_run_kind_idx"),
        ),
    ]
