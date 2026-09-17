# Frontend roadmap and ownership

Bu hujjat frontend uchun aniq backlog va UI ownership bo‘yicha single source of truth. Yangi funksiya qo‘shilganda avval shu yerda joyi, egasi va acceptance mezoni aniqlanadi.

## Product position

Quantum Circuit — local-first, reproducible quantum-circuit va tensor-network research workbench. Frontendning vazifasi marketing sahifa emas: muammoni tayyorlash, hisoblashni nazorat qilish, natijani tekshirish va qayta ishlab bo‘ladigan artifact olish.

## Hozir mavjud

- Circuit editor: gate placement, multi-qubit ko‘rinishi, validation, undo/redo.
- IR v1 va OpenQASM 3 import/export.
- Local/backend save-load.
- Run va Physics Lab workflow’lari.
- 1D/2D/3D lattice preview, Hamiltonian pluginlari.
- GPU agent status, async job progress va provenance.
- MPS/TEBD/DMRG/PEPS natijalarini ko‘rsatish.
- Noise, bond dimension, truncation va parameter sweep konfiguratsiyasi.

## P0 — platforma va UX poydevori

### 1. Markaziy design system

Owner: `apps/web/src/ui` va `apps/web/app/globals.css`

Acceptance:

- rang, spacing, radius, shadow, typography va status ranglari token sifatida bitta joyda bo‘ladi;
- `Button`, `Card`, `Metric`, `Field`, `ProgressBar` umumiy primitive sifatida ishlatiladi;
- sahifalar inline ranglar bilan yangi takroriy stil kiritmaydi;
- light workspace va dark Physics Lab bir xil semantic tokenlardan foydalanadi.

### 2. Umumiy application shell

Owner: `app/page.tsx`

Acceptance:

- navigation, agent status, active tab va responsive header bitta shell’da;
- keyboard focus, `aria-current`, disabled/loading state izchil;
- page-specific component shell layoutni qayta yaratmaydi.

### 3. Job lifecycle UI

Owner: `src/lib/agent.ts`, `src/ui`, `ResearchWorkspace`, `LatticeLab`

Acceptance:

- queued/running/done/failed/canceled holatlar ko‘rinadi;
- cancel, retry va error detail bitta umumiy paneldan ishlaydi;
- job tugaganda result va provenance yo‘qolmaydi.

## P1 — haqiqiy research workflow

### 4. Experiment history va replay

- har run konfiguratsiyasi, circuit version, seed, backend va artifact link bilan saqlanadi;
- Results sahifasida filter, replay va export mavjud bo‘ladi;
- bir xil input boshqa vaqtda qayta ishga tushirilganda diff ko‘rinadi.

Status: local-first v1 bajarildi. `runHistory.ts` request/result/provenance’ni local storage’da saqlaydi, Results filter/replay/JSON export beradi. Backend qatlamida point-level `Run`/`RunArtifact` va aggregate physics `Study` sync ishlaydi.

Backend sync adapter ham qo‘shildi: API mavjud bo‘lsa `Run`, `raw` `RunArtifact` va tugagan convergence campaign uchun `Study` yoziladi; API ishlamasa local-first workflow to‘xtamaydi.

### 5. Convergence va diagnostics

- DMRG: energy per sweep, residual, discarded weight;
- TEBD: `E(t)`, norm drift, bond growth, truncation error;
- PEPS: contraction status, approximation warning va boundary/method metadata;
- warning faqat bitta owner panelda ko‘rsatiladi, duplicate qilinmaydi.

Status: DMRG sweep va TEBD/PEPS trajectory diagnostics Results sahifasiga qo‘shildi; bounded DMRG/TEBD/PEPS convergence study ham qo‘shildi. Har bir nuqta alohida provenance va replayable history yozuviga ega, tugagan campaign esa backend `Study` manifestiga sinxronlanadi.

### 6. Parameter sweep analysis

- sweep jadvali bilan birga line/heatmap ko‘rinishi;
- failed point, valid point va uncertainty alohida belgilanadi;
- CSV/JSON artifact export qilinadi.

### 7. Compare workspace

- run A/B tanlash;
- backend, bond dimension, cutoff, energy, norm va runtime comparison;
- circuit yoki Hamiltonian o‘zgargan joylar diff sifatida ko‘rsatiladi.

Status: stored result A/B metric comparison v1 qo‘shildi; circuit/Hamiltonian structural diff va richer artifact diff keyingi iteratsiya.

## P2 — domain plugin layer

Har bir domain modul quyidagi contract bilan ulanadi: `manifest`, `input schema`, `preview`, `build problem`, `run adapters`, `result renderer`, `provenance metadata`.

- spin lattice: mavjud, polish qilinadi;
- Hubbard/materials: mavjud boshlang‘ich plugin, fermion mapping va observable UX kuchaytiriladi;
- fermion mapping: Jordan–Wigner/Bravyi–Kitaev natija ko‘rish va parity warning;
- chemistry/materials: basis/model import, sparse Hamiltonian preview, energy/ground-state workflow;
- keyingi modullar shu interface orqali qo‘shiladi, core editor va job panelga tegmaydi.

## P3 — scale va release quality

- katta circuit uchun viewport virtualization va LOD;
- artifact storage va offline replay;
- accessibility audit, responsive layout, error boundary va telemetry-free diagnostics;
- desktop/Tauri packaging va one-click agent lifecycle;
- lab/self-host deployment uchun shared project/run history.

## Ataylab qo‘shilmaydi

- marketing banner, hero section, dekorativ analytics;
- bir xil metric yoki warningni Inspector va Results’da to‘liq takrorlash;
- backend imkoniyati yo‘q bo‘lgan controlni UI’da “ishlaydigandek” ko‘rsatish;
- frontendda domain algoritmini ko‘paytirish — hisoblash agent/backend’da qoladi.

## Ishlash tartibi

1. Avval token va common primitive.
2. Keyin shared job/result contract.
3. Keyin page-specific research views.
4. Har modul uchun smoke test + production build.
5. Faqat acceptance mezoni bajarilgandan keyin keyingi domain plugin.
