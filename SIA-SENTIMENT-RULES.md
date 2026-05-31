# SIA-SENTIMENT-RULES.md — Pravila za sentiment optimizacijo

> Vir: Surfer SEO analiza 17.500 strani + empirično testirano na projektu megabazeni.si / salon OxM
> Datum: 2026-05-28

---

## 🎯 1. Pravi cilj sentimenta

**Cilj ni +0.50 — cilj je preseči +0.25.**

| Pas | Score | Status | Pojavljanje v SERP |
|---|---|---|---|
| Pozitivno | +0.25 do +1.0 | ✅ cilj | 87.71% strani |
| Nevtralno | -0.25 do +0.25 | ❌ najslabše | 0.26% strani |
| Negativno | -1.0 do -0.25 | ⚠️ odvisno od keywordov | 12.03% strani |

> Nevtralna stran je HUJŠA od negativne v smislu pojavnosti v SERPih. Nevtralno = neobstoječe.

---

## 🔍 2. Najprej preveri competitor sentiment

Preden določiš ciljni sentiment — preveri benchmark iz `analiza_*.md`:

- Konkurenti avg pozitivni → cilj: ujemi ali preseži njihovo povprečje
- Competitor SERP pozitivno dominiran → piši pozitivno
- Mešan SERP (pozitivno + negativno) → izberi eno stran, **ne sedi na sredini**

> Pravilo iz Surfer SEO članka: "Every SERP is different — analyze your competitors and find out what sticks to the top."

⚠️ Brez benchmark podatkov → ne določaj ciljnega score-a. Napiši samo "benchmark ni na voljo".

---

## ⛔ 3. "Ne" konstrukti so smrtonosni — pravilo zamenjave

Vsaka "ne X" konstrukcija zniža sentence score. **Vedno zamenjaj z "da Y".**

| ❌ Negativen frame | ✅ Pozitiven frame |
|---|---|
| "ne na make-upu" | "na zdravi, naravni koži" |
| "Rezultati, ki jih ne more skriti" | "Koža, ki seva sama od sebe" |
| "ki se oprimejo le dlačic in ne kože" | "koža ostane nedotaknjena in brez draženja" |
| "ne obtežijo trepalnic" | "trepalnice ostanejo lahke in naravne" |
| "brez bolečine in nič iritacij" | "prijetna in varna izkušnja" |
| "ne vsebujejo poceni polnil" | "visoka koncentracija aktivnih sestavin" |

> Dokazano iz analize: "ne more skriti" → -0.1 | "ne na make-upu" → 0.0 | "ne kože" → +0.1

**Kontrolno vprašanje pred oddajo:** Koliko "ne" je v besedilu? Vsak je potencialni score killer.

---

## 📋 4. Seznami storitev so vedno 0.00

Naštevanje storitev v enem stavku = vedno nevtralno. Google NLP ne zna oceniti "in/ali" listov.

❌ `"Nudimo klasično manikuro, biab/gel tehnike in permanentno lakiranje."` → 0.00

✅ Vsako storitev ločiti ali dodati outcome:
`"Biab in gel tehnike zagotavljajo trdne, zdrave nohte — obstojnost do 4 tedne."` → +0.30+

---

## ✍️ 5. Dokazano visoko-sentiment besedišče — Slovenian NLP

Besede in fraze, ki so v analizi dosegle +0.40 ali več:

| Beseda / fraza | Dokazani score | Opomba |
|---|---|---|
| "varno / varnost / varna" | +0.40–0.60 | deluje v kateremkoli kontekstu |
| "Seveda, ..." (začetek stavka) | +0.60 | samo če sledi pozitivna trditev |
| "niste omejeni" | +0.60 | svoboda = visok sentiment |
| "dobro počutje" | +0.50 | zelo zanesljivo |
| "prava investicija" | +0.40 | deluje za service/location strani |
| "zaupanja vreden / zaupate" | +0.40 | zaupanje = pozitivno |
| "sproščujoča spa izkušnja" | +0.40 | experiential language |
| "najvišjimi higienskimi standardi" | +0.40 | strokovnost = pozitivno |
| "si zaslužite" | testirati | obetavno |
| "zadovoljni / zadovoljstvo" | testirati | obetavno |

---

## ➕ 6. Sentiment se ZAMENJA, ne dodaja

**Dodajanje novih stavkov razredči povprečje.**

Z vsakim novim nevtralnim stavkom se score spušča, ne dviga.

Pravilo: **1 out, 1 in** — za vsak odstranjeni nevtralni stavek vstavi pozitivnega na isto mesto.

> Primer iz prakse: stran s 68 stavki pri +0.20. Dodali smo 10 pozitivnih stavkov → score šel na samo +0.21. Nič se ni premaknilo.
> Ko smo začeli ZAMENJEVATI → score se je premaknil.

---

## 🏷️ 7. Heading vpliva na score celotne sekcije

Heading ki vsebuje "ne" ali negativno framing → potegne score celotnega združenega stavka navzdol.

❌ `"Rezultati, ki jih ne more skriti niti jutro brez ličil"` → -0.1 za celo sekcijo
✅ `"Koža, ki seva sama od sebe"` → +0.30+

**Pravilo:** Vsak H2/H3 heading preveri za besede: `ne`, `brez`, `proti`, `problem`, `težava`, `preprečiti`.

---

## 📏 8. Realni cilji glede na obseg dela

| Cilj | Zahtevan obseg | Pristop |
|---|---|---|
| **+0.25** (pozitivno) | Minimalen — 30 min | Popravi "ne" frame + 2–3 key stavke |
| **+0.30–0.35** | Zmeren — 1–2 uri | Rewrite vseh 0.00 stavkov + headingi |
| **+0.40+** | Večji — pol dneva | Skrajšaj stran + rewrite opisov |
| **+0.50+** | Strukturna sprememba | Skrajšaj na ~20 stavkov, samo visoko-sentiment vsebina |

---

## 🔄 9. Delovni postopek za sentiment optimizacijo

```
1. Preberi analiza_*.md → kakšen je current score?
2. Je score pod +0.25? → PRIORITETA: preseči +0.25
3. Preveri competitor avg iz benchmarka → določi realni cilj
4. Identificiraj vse stavke z "ne" → zamenjaj z "da Y"
5. Identificiraj vse 0.00 stavke (seznami storitev) → rewrite z outcome
6. Preveri vse H2/H3 headinge za negativne besede → zamenjaj
7. Zamenjuj, ne dodajaj — 1 out, 1 in
8. Testiraj z analizo → ali smo presegli +0.25?
```

---

## ⚠️ Opozorilo za slovenščino

Google NLP za slovenščino ima omejitve. Sentiment score in magnitude sta zanesljiva, sentence-level razčlenitev pa je včasih nepredvidljiva.

Zato: **vedno testiraj z analizo po spremembi**, ne predpostavljaj score na podlagi "zveni pozitivno".

---

*Generirano: 2026-05-28 | Vir: Surfer SEO + empirična analiza OxM salon Ljubljana*
