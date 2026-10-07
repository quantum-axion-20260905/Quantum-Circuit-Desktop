# Quantum Circuit Desktop — Modular & Layered Architecture Upgrade

Ushbu hujjat loyihada amalga oshirilgan **qatlamli (layered)**, **modulli (modular)** va **plaginli (plugin-based)** arxitektura yangilanishini tushuntiradi.

---

## 1. Arxitektura Tuzilmasi (Architectural Layers)

Yangi arxitektura qat'iy javobgarlik chegaralari (Separation of Concerns) asosida tashkil etilgan:

```mermaid
flowchart TD
    subgraph UI_Presentation [1. UI & Desktop Shell Layer]
        NextApp[Next.js 16 Web App]
        TauriShell[Tauri 2 Rust Desktop Shell]
        ReactFlowEditor[Circuit & Lattice Visualizers]
    end

    subgraph API_Gateway [2. Modular API Routing Layer]
        AppFactory[FastAPI Root / Application Factory]
        SystemRouter[system.py: /health, /hardware, /capabilities, /metrics]
        CircuitsRouter[circuits.py: /jobs/run, /preflight, /sweep, /simulate]
        PhysicsRouter[physics.py: /jobs/dmrg, /ctmrg, /peps, /tebd, /expectation]
        AsyncJobsRouter[async_jobs.py: /async/jobs, /async/{kind}, cancellation]
        PluginsRouter[plugins.py: /plugins/{id}/*]
    end

    subgraph Service_Orchestration [3. Services & Orchestration Layer]
        DispatchService[dispatch.py: async parser, backend resolver, job lifecycle]
        CommonService[common.py: resource broker, GPU guard, limits, telemetry]
        JobManager[JobManager: thread queue, durable journal, device leases]
    end

    subgraph Domain_Plugins [4. Extensible Domain Plugins Layer]
        BasePlugin[BaseDomainPlugin Protocol & Class]
        SpinLattice[Spin-Lattice: Ising, Heisenberg, XXZ]
        Hubbard[Hubbard Materials & Jordan-Wigner Fermions]
        CustomPlugins[Dynamic Entry Points & Custom Plugins]
    end

    subgraph Core_Numerics [5. Pure Mathematical & Compute Kernels]
        MPS[MPS / DMRG / TEBD Runtime]
        PEPS[2D PEPS & Boundary-MPS Contraction]
        CTMRG[CTMRG Dynamic & Gauge Covariance Kernels]
        Reference[Deterministic Reference CPU Simulator]
        CuPyGPU[CUDA / CuPy Statevector Kernels]
    end

    UI_Presentation --> API_Gateway
    API_Gateway --> Service_Orchestration
    Service_Orchestration --> Domain_Plugins
    Service_Orchestration --> Core_Numerics
```

---

## 2. Nimalar O‘zgartirildi va Yaxshilandi?

1. **`server.py` monolitidan qutulindi:**
   - Oldin: ~1900 qatorli ulkan monolit fayl bo‘lib, routing, biznes logika, runnerlar va validatsiyalar aralashib ketgan edi.
   - Hozir: Atigi ~180 qatorli toza **Application Factory**. U faqat middleware-larni ulaydi va modulli routerlarni mount qiladi.
   
2. **Modulli API Routerlar (`agent/qc_agent/api/routers/`):**
   - `system.py`: Server holati, texnik ko‘rsatkichlar, apparat vositalari, imkoniyatlar katalogi.
   - `circuits.py`: Kvant sxemalarini hisoblash, preflight xarajat tahlili, parametrlar sweepi, tekshiruvlar.
   - `physics.py`: DMRG, CTMRG (dinamik, boundary-MPS, gauge kovarianti), PEPS, TEBD va energetik kutilmalar.
   - `async_jobs.py`: Uzoq davom etuvchi vazifalarni fon rejimida bajarish, navbat va bekor qilish.

3. **Xizmatlar Qatlami (`agent/qc_agent/services/`):**
   - `services/common.py`: GPU resurslarini bron qilish (`_sync_gpu_guard`), xotira limitlari va audit jurnallari.
   - `services/dispatch.py`: Asinxron marshrutlash, dinamik tahlil va progress monitoringi.

4. **Kengaytiriluvchi Plaginlar Tizimi (`agent/qc_agent/plugins/`):**
   - `BaseDomainPlugin` klassi qo‘shildi.
   - Yangi plagin yaratish uchun `plugins/template.py` namunasi berildi.
   - Plaginlar Python `entry_points` orqali dinamik kashf etiladi va mustaqil ishlaydi.

5. **100% Orqaga Moslik (Backward Compatibility):**
   - Barcha 203 ta unit va integratsiya testlar (194 passed, 9 GPU skipped, 0 failed) to‘liq muvaffaqiyatli o‘tdi.
   - Eski importlar (`from qc_agent.server import ...`) to‘liq qo‘llab-quvvatlanadi.

---

## 3. Yangi Feature yoki Fizika Modelini Qanday Qo‘shish Mumkin?

Yangi model (masalan, Kvant Kimyosi VQE yoki QAOA optimizatori) qo‘shish endi juda oson:

```python
from qc_agent.plugins.base import BaseDomainPlugin, PluginInfo
from qc_agent.plugins.registry import register

class MyQuantumChemistryPlugin(BaseDomainPlugin):
    info = PluginInfo(
        id="quantum-chemistry",
        name="Molecular Chemistry VQE",
        version="1.0.0",
        description="Generates molecular orbital Hamiltonians",
        capabilities=("chemistry", "vqe"),
    )

    def build_hamiltonian(self, payload):
        # O'z algoritmingizni yozasiz
        return {"terms": [...]}

# Ro'yxatdan o'tkazish:
register(MyQuantumChemistryPlugin())
```

Hech qanday asosiy server yoki dvigatel kodiga teginish shart emas!
