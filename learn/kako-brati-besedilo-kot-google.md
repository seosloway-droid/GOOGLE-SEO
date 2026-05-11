# Kako brati besedilo kot Google NLP

> Cilj: Ko prebereš katerokoli besedilo, v 2–3 minutah veš, katero besedo Google vidi kot glavno temo, katere entitete "kradejo" salience in kaj bi bilo treba popraviti.

---

## Osnova: Kako Google NLP ocenjuje besedilo

Google ne bere besedila kot človek. Razčleni ga na **entitete** in vsakemu dodeli:

- **Salience** — kako pomembna je ta entiteta za celotno besedilo (0–100%)
- **Tip** — PERSON, ORGANIZATION, LOCATION, OTHER, CONSUMER_GOOD...
- **Knowledge Graph** — ali entiteto pozna iz svojega globalnega znanja

Salience ni samo število pojavitev. Google jo izračuna iz treh faktorjev:

1. **Pozicija** — besede na začetku besedila dobijo višjo salience. Prva 100 besed je kritičnih.
2. **Grammatična vloga** — subjekt stavka > objekt > stranska omemba
3. **Število omemb** — a šele na tretjem mestu

**Primer:** Beseda omenjena 1x kot subjekt v drugem stavku lahko dobi višjo salience kot beseda omenjena 5x v stranskih zvezah na koncu besedila.

---

## Korak 1: Nauči se videti entitete

Ko bereš besedilo, si mentalno (ali s svinčnikom) označi:

- Vsako **lastno ime** (Intex, Megabazeni, Easy Set)
- Vsako **blagovno znamko** ali **akronim** (XTR, PVC, pH)
- Vsak **tehnični termin** (peščeni filter, galvanizirani okvir)
- Vsako **mersko enoto ali število z enoto** (7,2 pH, 15 min, 35 m²)
- Vsak **specifičen produkt** (Power Steel, Ultra XTR Frame)

Vse to so potencialne entitete. **Več kot jih je, bolj je salience razpršena** — nobena ne dobi visokega deleža.

### Rdeča zastavica: Nenamerne entitete

Posebej pazi na:
- **Simbole** (™, ®, ©) — Google jih prereže besedo in ustvari lažno entiteto
- **Kratice in akronime** — dobijo visoko salience ker izgledajo "unikatno"
- **Dolge fraze** — Google jih včasih vzame skupaj kot eno entiteto

**Primer iz prakse:** `Ultra XTR Frame™ tehnologijo` → Google prebere "TM tehnologijo" kot entiteto s 5.1% salience, ker ™ prereže frazo. Ena nenamerna entiteta, ki ukrade salience od ključne besede.

---

## Korak 2: Nauči se videti subjekte

Za vsak stavek si zastavi eno vprašanje: **"Kdo ali kaj izvaja dejanje?"**

To je subjekt. Subjekt dobi višjo salience od vseh ostalih besed v istem stavku.

| Stavek | Subjekt (visoka salience) | Objekt (nižja salience) |
|---|---|---|
| "Intex ponuja bazene za vse" | Intex | bazene |
| "Bazeni Intex so globalni standard" | Bazeni Intex | / |
| "Na Megabazeni.si najdete bazene" | ti (implicirano) | bazene |

**Pravilo:** Če hočeš, da Google vidi besedo X kot glavno temo — naredi X subjekt čim večih stavkov.

---

## Korak 3: Trije preprosti testi pri branju

Ko prebereš katerokoli besedilo, si zastavi te tri vprašanja:

**Test 1 — Kdo je subjekt?**
Preberi prvih 5 stavkov. Katera beseda je najpogostejši subjekt? To je verjetno entiteta z najvišjo salience. Je to ključna beseda, ki jo želiš rankat?

**Test 2 — Koliko entitet tekmuje?**
Preštej: koliko različnih znamk, produktov, tehničnih terminov, merskih enot je v besedilu? Če je odgovor "veliko" — salience je razpršena in nobena ključna beseda ne bo dominirala.

**Test 3 — Kje se ključna beseda pojavi prvič?**
Je v prvem stavku? V prvem odstavku? Ali šele v tretjem odstavku? Pozicija prve omembe močno vpliva na salience.

---

## Korak 4: Beri konkurenčna besedila drugače

Ko bereš opis kategorije ali članek konkurenta, ne bereš vsebine — bereš **strukturo**.

Vprašaj se:
- Katera beseda je subjekt v večini stavkov?
- Koliko entitet tekmuje za salience?
- Je ključna beseda visoko ali nizko v besedilu?
- So v besedilu simboli, akronimi, tehični termini, ki ustvarjajo nenamerne entitete?

---

## Vadba: En tekst na dan, 5 minut

1. Vzemi katerokoli besedilo (opis kategorije, članek, oglas)
2. Z rdečo podčrtaj vse entitete (lastna imena, znamke, tehnični termini)
3. Z modro podčrtaj subjekte vsakega stavka
4. Oceni: "Katera beseda ima največjo salience? Je to ključna beseda?"
5. Prilepi v Google NLP demo in preveri svojo oceno

**Google NLP demo:** https://cloud.google.com/natural-language — klikni "Try the API" → Entities

Po 2–3 tednih (en tekst na dan) začneš to videti avtomatsko, brez orodja.

---

## Hitri referenčni seznam: Kaj dvigne salience

| Akcija | Učinek |
|---|---|
| Ključna beseda kot subjekt stavka | ↑↑ Visok učinek |
| Ključna beseda v prvih 100 besedah | ↑↑ Visok učinek |
| Ključna beseda v H1 in H2 naslovih | ↑ Srednji učinek |
| Manj konkurenčnih entitet (znamke, akronimi, merske enote) | ↑ Srednji učinek |
| Večkratne omembe ključne besede | ↑ Nizek učinek (sam po sebi) |
| Odstranitev simbolov ™ ® © | ↑ Prepreči "krajo" salience |

## Kaj znižuje salience

| Akcija | Učinek |
|---|---|
| Ključna beseda samo kot objekt ali v stranski zvezi | ↓ |
| Preveč znamk, akronimov, tehničnih terminov | ↓↓ |
| Ključna beseda prvič omenjena šele v sredini besedila | ↓ |
| Simboli ™ ® ki ustvarijo lažne entitete | ↓↓ |

---

*Zapisano: maj 2026 — na podlagi analize bazeni_20260510*
