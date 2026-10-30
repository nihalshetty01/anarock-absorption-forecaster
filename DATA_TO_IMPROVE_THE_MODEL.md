# Data That Would Make the Model More Accurate

**Principle:** accuracy comes from better information about each project and from real outcomes, not from more synthetic rows.

## A. Real outcome data (most important: replaces synthetic data)
| # | Data / document | What it gives | Source | Access |
|---|---|---|---|---|
| 1 | ANAROCK's own launch and absorption database | Real 12-month sales by project and configuration, the true target | ANAROCK Research (internal) | **Internal only.** This is the pitch: "plug in your data" |
| 2 | MahaRERA Quarterly Progress Reports (QPRs) | Units booked per quarter → absorption over time | maharera.maharashtra.gov.in | Public; slow, self-reported, ~62% compliance |
| 3 | Developer investor presentations and quarterly results | Launch-quarter sales ("sold X units / ₹Y cr in launch") | Godrej, Kolte-Patil, others' investor pages | Public, listed developers only |
| 4 | Property registration data (IGR Maharashtra) | Actual sale registrations by project and date | igrmaharashtra.gov.in | Public; needs project matching |
| 5 | ANAROCK published quarterly city reports | Market-level Pune sales, launches, unsold stock → market trend input | anarock.com research | Public |

## B. Better information about each project (new inputs)
| # | Input | Why it helps | Source |
|---|---|---|---|
| 6 | Exact location (pin code or lat-long) | Fixes the "fringe location looks cheap" error (the Ashiana case) | RERA registration, developer site |
| 7 | Local price baseline within 2 km | Fairer "price vs market" comparison | Registrations (IGR), ANAROCK pricing data |
| 8 | Amenities and project size (land acres, open space) | Captures project quality | RERA project details, brochures |
| 9 | Launch offers and payment plans (e.g. 20:80) | Strong short-term sales driver | Developer marketing, channel partners |
| 10 | Pre-launch demand: enquiries, site visits, EOIs (expressions of interest) | Best early signal of absorption | ANAROCK CRM / channel-partner data (internal) |
| 11 | Competing supply launched nearby in the same quarter | Competition dilutes absorption | ANAROCK supply data, RERA new registrations |
| 12 | Infrastructure timeline (metro line opening, new IT park) | Future demand driver | PMRDA / Pune Metro announcements, news |
| 13 | Home-loan rates at launch | Affordability driver | RBI repo rate history |

## C. Leakage warning (for the interview)
Only use what is known **on launch day**. Site visits *after* launch, discounts given later, or "units sold in month 3" would leak the answer into the inputs. They make the model look great in testing and fail in real use.

## D. Priority if ANAROCK gave you 3 months
1. Item 1 (real target) → retrain and compare against this prototype.
2. Items 6, 7 (exact location, local price) → fixes the biggest known error.
3. Item 10 (pre-launch demand) → biggest accuracy gain, and ANAROCK already has it through its broker network.
