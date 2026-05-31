from rest_framework.routers import DefaultRouter

from .views import CircuitVersionViewSet, ProjectViewSet, RunArtifactViewSet, RunViewSet

router = DefaultRouter()
router.register(r"projects", ProjectViewSet)
router.register(r"versions", CircuitVersionViewSet)
router.register(r"runs", RunViewSet)
router.register(r"artifacts", RunArtifactViewSet)

urlpatterns = router.urls
