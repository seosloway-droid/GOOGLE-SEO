# SEO Content Project — megabazeni.si

## ⚠️ PRED VSAKIM DELOM — OBVEZNO

1. Preberi `/AGENT.md` — tam je **PRE-FLIGHT CHECKLIST** ki ga moraš izvesti pred vsako nalogo
2. Preberi `/WRITING-STYLE.md` — ton, stil, struktura pisanja
3. Poisči najnovejše datoteke v `analize/` mapi

**Ne začenjaj nobene naloge brez teh korakov.**
Če kateri podatek manjka → napiši `[CRITICAL_PAUSE]` in pojasni kaj potrebuješ.

---

## Dodatne reference (preberi ko pišeš vsebino)

| Datoteka | Namen |
|---|---|
| `/SIA-ONPAGE-RULES.md` | Empirično testirani SEO faktorji — keyword placement, H1/H2 pravila, density, interni linki |
| `/SIA-ARTICLE-STRUCTURE.md` | Template za strukturo članka — H1/H2/H3 hierarhija, keyword razporeditev, checklist |
| `/SIA-SENTIMENT-RULES.md` | Pravila za sentiment optimizacijo — cilji, "ne" framing, visoko-sentiment besedišče, delovni postopek |

---

# Datoteke v tej mapi

## analiza_{keyword}_{timestamp}.md
SEO NLP analiza generirana iz appa. Vsebuje:
- Sentiment score + magnitude
- Entitete + salience
- Sentence-level sentiment (negativni stavki)
- Syntax (lexical density, pasivni glas)
- Entity sentiment
- Benchmark vs konkurenti (če je bil izveden)
- AI SEO Coach report (če je bil generiran)

## content_brief_{keyword}_{timestamp}.md
Content brief generiran iz benchmark analize konkurentov. Vsebuje:
- Ciljna dolžina in struktura
- Obvezne entitete s ciljno salience
- Predlagana struktura H2 naslovov
- Ciljne NLP metrike
- People Also Ask vprašanja

## plan_izboljsav_{keyword}_{timestamp}.md
Plan izboljšav za obstoječe besedilo. Vsebuje:
- Kaj dodati (konkretni stavki)
- Kaj spremeniti (pred → po)
- Kaj odstraniti
- Manjkajoče entitete
- NeuroWriter term coverage

---

# Kako delati s temi datotekami

## "check analizo" ali "priporocaj kako naprej"
1. Najdi najnovejšo `analiza_*.md` datoteko
2. Preberi celotno analizo
3. Preberi AGENT.md za pravila glede tipa strani
4. Poda:
   - **Top 3 prioritetne akcije** po SEO vplivu
   - **Konkretne predloge za prepisovanje** za največje probleme
   - **Content gap** — katere entitete/teme manjkajo
   - **Ciljne vrednosti** — sentiment, lexical density, salience
   - **Quick wins** — kaj popraviti v manj kot 1 uri

## "izboljšaj besedilo" ali "prepiši po planu"
1. Najdi `plan_izboljsav_*.md` ali `analiza_*.md`
2. Preberi WRITING-STYLE.md za ton in stil
3. Preberi AGENT.md za tip strani in NLP cilje
4. Izvedi spremembe po prioritetnem vrstnem redu iz plana
5. Za slovenščino: ne zanašaj se na Google NLP verb/adjective counts — ti so nezanesljivi

## "napiši po briefu"
1. Najdi `content_brief_*.md`
2. Preberi WRITING-STYLE.md
3. Piši točno po briefu — dolžina, naslovi, entitete, ton

---

# Konvencija imen datotek

| Datoteka | Kdaj nastane |
|---|---|
| `analiza_{kw}_{ts}.md` | Po analizi strani v appu |
| `content_brief_{kw}_{ts}.md` | Po "Pišem novo" v Content Brief strani |
| `plan_izboljsav_{kw}_{ts}.md` | Po "Izboljšujem obstoječe" v Content Brief strani |

---

# Legenda v poročilih

- 🔵 Uradni Google podatek — direktno iz Google NLP API
- 🟠 SEO best practice — industrijski standard, ni uradno potrjeno od Googla
- 🟣 Google guidelines — iz Google Quality Rater Guidelines

---

# Opomba za slovenščino

Google NLP API **ne podpira** slovenščine za:
- Entity sentiment (ni na voljo)
- Syntax/POS tagging (nezanesljivo — verb count je pogosto napačen)

Google NLP API **podpira** za slovenščino:
- Sentiment score + magnitude ✅
- Entity names + salience ✅
- Content categories ✅

Za slovensko sintaksno analizo app uporablja Codex AI — zanesljivost ~85%.

## Imported Claude Cowork project instructions
