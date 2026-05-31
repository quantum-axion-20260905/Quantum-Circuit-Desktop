# UI Layout Contract (No Duplication)

## Principle
- Har bir ma'lumot **faqat bitta asosiy joyda** bo'ladi.
- `Inspector` = kontekst xulosasi (qisqa).
- `Bottom tabs` = ishlash maydoni (batafsil).

## Ownership
- **Center Canvas (Main page area)**  
  - `Design`: circuit chizish/tahrirlash  
  - `Run/Results/Experiments`: workflow konteksti va status

- **Right Panel: Inspector only**  
  - `Design`: circuit sanity + qisqa statistik summary  
  - `Run`: joriy run snapshot  
  - `Results`: joriy natija snapshot  
  - Bu panelda `Metrics/Warnings` bo'lmaydi (duplicate oldini olish uchun).

- **Bottom Panel: Detailed work tabs**
  - `Workbench`: page-specific asosiy amallar (`DesignWorkbench` yoki `ExplainPanel`)
  - `Metrics`: barcha metrikalar (circuit + latest run)
  - `Warnings`: barcha ogohlantirishlar
  - `Compare`: run A/B taqqoslash
  - `Notes`: tadqiqot qaydlari

## Rules
- Yangi tab qo‘shilsa, avval owner aniqlanadi (`Inspector` yoki `Bottom`), ikkalasiga birdan emas.
- Bir xil JSON/snapshot bir nechta panelda to‘liq ko‘chirilmaydi.
- `Inspector` doim qisqa, qaror uchun; tahlil `Bottom` tablarda.
