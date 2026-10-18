# Quantum Circuit Platform — Domain Architecture & Functional Specifications

Ushbu hujjat **Quantum Circuit Desktop** platformasining sohaviy arxitekturasi, xizmat ko‘rsatish sohalari (domains), maqsadli foydalanuvchilar va har bir modulning funksional vazifalarini belgilaydi.

---

## 1. Umumiy Konseptsiya

Platforma kvant hisoblash, nazariy fizika va hisoblash materialshunosligini yagona, zamonaviy va unumdor desktop ish stoli (Workstation/IDE) muhitida birlashtirishga mo‘ljallangan. 

Loyiha 4 ta asosiy mustaqil sohaga (Domains) bo‘lingan:
1. **Kvant Sxemalari Muhandisligi va Algoritmlar (Circuit Engineering Studio)**
2. **Kvant Ko‘p Zarrali Fizikasi va Materialshunoslik (Many-Body Physics & Materials Studio)**
3. **Yuqori Unumdorlikdagi Kvant Simulyatsiyasi (High-Performance Quantum Simulation)**
4. **Kvant Ta’limi va Interaktiv Vizualizatsiya (Interactive Quantum Lab / Quirk-style Playground)**

```
+--------------------------------------------------------------------------------------------------+
|                                    QUANTUM CIRCUIT WORKBENCH                                     |
+--------------------------------------------------------------------------------------------------+
|  Circuit Studio       |  Physics & Materials  |  Interactive Lab     |  Simulator & Run  | History|
+-----------------------+-----------------------+----------------------+-------------------+--------+
| - OpenQASM 3.0 AST    | - 1D/2D/3D Lattices   | - Live Bloch Spheres | - GPU Statevector | SQLite |
| - Gate Composer Bar   | - Spin & Hubbard      | - Amplitude Phasors  | - Tensor MPS/TN   | Proven-|
| - Presets Library     | - DMRG / TEBD / PEPS  | - Zero-latency probe | - Noise Models    |  ance  |
| - LaTeX & CSV Export  | - Correlation Heatmap | - Educational Tools  | - Live Histogram  | Archive|
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Sohalar Tavsifi (Domains Specification)

### 1-Soha: Kvant Sxemalari Muhandisligi va Algoritmlar (Circuit Engineering Studio)
* **Maqsad:** Kvant algoritmlarini loyihalash, sintaktik tekshirish va xalqaro standartlarga eksport qilish.
* **Maqsadli Auditoriya:** Kvant dasturchilari, kvant axborot nazariyasi tadqiqotchilari, algoritmlar muhandislari.
* **Asosiy Funksionallik:**
  * **Desktop Menubar:** Professional CAD/IDE menyulari (File, Edit, Presets, Export, View).
  * **Gate Composer Palette:** $H, X, Y, Z, S, T, R_x, R_y, R_z, CX, CZ, SWAP$ geytlari palitrasi.
  * **Kvant Algoritmlari Kutubxonasi:** Bell State ($|\Phi^+\rangle$), GHZ (3-qubit), Superposition, Quantum Teleportation, Quantum Fourier Transform (QFT), Grover Search, Quantum Phase Estimation (QPE).
  * **Ko‘p Formatli Eksport:**
    * OpenQASM 3.0 (`.qasm`)
    * LaTeX Quantikz2 (`.tex` — ilmiy maqolalar uchun)
    * Circuit CSV Matrix (`.csv`)
    * Native Quantum IR AST (`.json`)

---

### 2-Soha: Kvant Ko‘p Zarrali Fizikasi va Materialshunoslik (Many-Body Physics & Materials Studio)
* **Maqsad:** Kondensirlangan modda fizikasi, qattiq jismlar kvant nazariyasi, magnit materiallar, o‘ta o‘tkazuvchanlik va topologik fazaviy o‘tishlarni modellashtirish.
* **Maqsadli Auditoriya:** Kondensirlangan modda fiziklari, materialshunoslar, kvant kimyogarlari, nazariy fizika kafedralari.
* **Asosiy Modellar va Fizik Tizimlar:**
  * **Spin Panjaralari (Spin Lattice Models):**
    * *Transverse Field Ising Model:* $H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i$ (Kvant ferromagnetizm va fazaviy o‘tishlar).
    * *Heisenberg Model:* $H = J \sum_{\langle i,j \rangle} (X_i X_j + Y_i Y_j + Z_i Z_j)$ (Kvant antiferromagnetizm).
    * *XXZ Anisotropic Model:* $H = \sum_{\langle i,j \rangle} [J(X_i X_j + Y_i Y_j) + \Delta Z_i Z_j] - h \sum_i Z_i$.
  * **Elektron Materiallar (Fermionic Materials):**
    * *Fermi-Hubbard Model:* $H = -t \sum_{\langle i,j \rangle, \sigma} (c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.}) + U \sum_i n_{i\uparrow} n_{i\downarrow} - \mu \sum_i n_i$ (Kuchli korrelyatsiyalangan elektronlar, Mott o‘tishi, yuqori haroratli o‘ta o‘tkazuvchanlik).
* **Hisoblash va Tensiyel Tarmoq Dvigatellari:**
  * **DMRG (Density Matrix Renormalization Group):** 1D va 2D zanjirlar uchun minimal asosiy holat energiyasi (Ground State Energy).
  * **Exact Diagonalization (ED):** Kichik tizimlar ($N \le 12$) uchun aniq tekshiruv.
  * **TEBD (Time-Evolving Block Decimation):** 1D zanjirlarda real vaqtli kvant dinamikasi ($e^{-i H t}$).
  * **Native PEPS (Projected Entangled Pair States):** 2D/3D panjaralarda fazoviy tensiyel tarmoq qisqarishi.
  * **CTMRG (Corner Transfer Matrix Renormalization Group):** Cheksiz 2D termodinamik limit ($N \to \infty$) uchun erkin energiya va magnitlanish.
* **Materialshunoslik Tahlili:**
  * Saytlararo korrelyatsiya funksiyalari $\langle S_i^z S_j^z \rangle$.
  * Mahalliy magnitlanish profili $\langle S_i^z \rangle$ va elektron zichligi $\langle n_i \rangle$.
  * Kvant fazalar tasnifi (Paramagnet, Ferromagnet, Antiferromagnet, Mott Insulator).

---

### 3-Soha: Yuqori Unumdorlikdagi Kvant Simulyatsiyasi (High-Performance Simulation & Hardware Profiling)
* **Maqsad:** Kvant sxemalarini real apparatga (QPU) yuborishdan oldin GPU va gibrid tizimlarda yuqori aniqlikda simulyatsiya qilish, xatoliklarni modellashtirish.
* **Maqsadli Auditoriya:** Kvant muhandislari, simulyatsiya laboratoriyalari, NISQ algoritmlari tadqiqotchilari.
* **Asosiy Imkoniyatlar:**
  * **Bir bosishda to‘liq ijro (One-Click Execution):** `▶ Run Simulation`.
  * **Hisoblash Backend lari:** GPU Statevector (Exact), GPU Tensor Network (MPS), Reference CPU.
  * **Haqiqiy Shovqin Modellari (Realistic Noise):** 1-qubit depolarizing, 2-qubit depolarizing, readout measurement flip.
  * **Preflight Feasibility tahlili:** Maksimal RAM xotira ehtiyoji, tensiyel qisqarish murakkabligi va vaqt hisobi.
  * **O‘lchov Natijalari Gistogrammasi:** Dominant holatlar, o‘lchangan holatlarning to‘liq ehtimollik taqsimoti, amplitudes va hisoblash telemetriyasi.

---

### 4-Soha: Kvant Ta’limi va Interaktiv Vizualizatsiya (Interactive Quantum Lab / Quirk-style Playground)
* **Maqsad:** Kvant mexanikasining murakkab prinsiplarini (superpozitsiya, chalkashlik, fazalar, interferensiya) interaktiv, nol-kechikishli vizual datchiklar bilan o‘rgatish.
* **Maqsadli Auditoriya:** Universitet talabalari, professor-o‘qituvchilar, kvant fizikasiga qiziquvchilar.
* **Asosiy Imkoniyatlar (Kelgusi Bosqich):**
  * **Real-Time Interactive Bloch Sphere:** Har bir qubit holatining 3D fazodagi yo‘nalishi.
  * **Phase & Amplitude Dial (Fazorlar):** Amplituda va faza burchaklarini soat mili kabi ko‘rsatuvchi doiralar.
  * **Inline State Probes:** Simlar ustida to‘g‘ridan-to‘g‘ri o‘lchov ehtimolliklarini ko‘rsatish.

---

## 3. Texnologik Stek va Integratsiya

* **Desktop Yadro:** Rust (Tauri v2) + SQLite (Local-first offline storage)
* **Foydalanuvchi Interfeysi:** Next.js (React 19 / Turbopack) + Full-Bleed CAD/IDE Styling
* **Hisoblash Agenti (Compute Core):** Python 3.12 + FastAPI + PyTorch + opt_einsum + Cotengra + cuQuantum (ixtiyoriy GPU tezlatkich).
