# Smoke-Test Report — 50 Gold-Standard Queries

Sequential run of the 50-query curated smoke-test suite against the full agentic Census pipeline.  The suite exercises every major feature — Tier 1 curated concepts, semantic-router fallback, composite neighborhoods, trend and comparator expansion, disparity framings, grant-narrative frame expansion, peer retrieval, scope-gate refusal, and exclusion-compound handling.

## Aggregate results

- **Total queries:** 50
- **Completed without error:** 50/50 (100%)
- **Returned prose:** 50/50 (100%)
- **Mean rating:** 9.3/10
- **Median rating:** 10/10
- **Total suite runtime:** 19.1 minutes

### Rating histogram

| Rating | Count | Bar |
|---|---|---|
| 10 | 40 | ████████████████████████████████████████ |
| 9 | 3 | ███ |
| 8 | 0 |  |
| 7 | 1 | █ |
| 6 | 2 | ██ |
| 5 | 4 | ████ |
| 4 | 0 |  |
| 3 | 0 |  |
| 2 | 0 |  |
| 1 | 0 |  |

## Per-query results

### Q01 — L1 simple lookup

**Query:** What's the median household income in DeKalb County, Georgia?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 10.8s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260422_235937_218093`

**Response prose:**

> In 2024, the median household income in DeKalb County, Georgia was $80,644. This figure is roughly in line with the metro level ($92,344) and the national median of $80,734, while slightly higher than the Georgia state median of $79,991. The median household income in DeKalb County has shown a strong rising trend, increasing by 16.2% from $69,423 in 2021.

**Key findings:**
- DeKalb County's median household income was $80,644 in 2024.
- This is comparable to the national median ($80,734) and slightly above the state median ($79,991).
- The county's median income is somewhat lower than the Atlanta-Sandy Springs-Alpharetta, GA Metro Area median of $92,344.
- From 2021 to 2024, the median household income in DeKalb County increased by 16.2%.
- Try next: How has median household income changed here recently?
- Try next: What is the poverty rate in this area?
- Try next: How does income vary across different racial or ethnic groups here?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B19013`
- `acs/acs5/2021/B19013`
- `acs/acs5/2024/B19013`

**Peer comparisons:**
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia

---

### Q02 — L1 simple lookup

**Query:** What's the poverty rate in Fulton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 7.9s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260422_235958_057325`

**Response prose:**

> In 2024, the poverty rate in Fulton County was 12.3%. This figure is somewhat above the Atlanta-Sandy Springs-Alpharetta, GA Metro Area rate of 10.0%, but slightly below the state of Georgia's rate of 12.6% and the national rate of 12.5%. Compared to 2021, when the poverty rate was 12.9%, the 2024 rate represents a mild decrease of 4.0%.

**Key findings:**
- Fulton County's poverty rate in 2024 was 12.3%.
- This rate is somewhat above the metro area's rate (10.0%).
- The county's poverty rate is slightly below both the state (12.6%) and national (12.5%) averages.
- The poverty rate in Fulton County has seen a mild decrease of 4.0% since 2021.
- Try next: How has the poverty rate changed here recently?
- Try next: What is the child poverty rate in this area?
- Try next: How does the poverty rate vary by race or ethnicity?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2024/B17001`

**Peer comparisons:**
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia

---

### Q03 — L1 simple lookup

**Query:** What's the unemployment rate in Clayton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 9.1s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000015_951334`

**Response prose:**

> In 2024, the unemployment rate in Clayton County was 7.2%. This rate is substantially above the Atlanta-Sandy Springs-Alpharetta, GA Metro Area level of 4.5%, the Georgia state level of 4.7%, and the national average of 5.2%. The 2024 rate for Clayton County is 1.58 times the metro average, indicating it is far above the expected range. This also represents a mild increase of 3.4% from the 2021 rate of 7.0%.

**Key findings:**
- Clayton County's unemployment rate in 2024 was 7.2%.
- This rate is substantially higher than the metro area (4.5%), state (4.7%), and national (5.2%) averages.
- The 2024 unemployment rate in Clayton County is 1.58 times the metro average.
- The unemployment rate in Clayton County has risen mildly by 3.4% since 2021.
- Try next: How has the unemployment rate changed here recently?
- Try next: How does the unemployment rate here compare to the state average?
- Try next: What are the unemployment rates for different racial or ethnic groups?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B23025`
- `acs/acs5/2021/B23025`
- `acs/acs5/2024/B23025`

**Peer comparisons:**
- *economic* (within state 13): Richmond County, Georgia, Henry County, Georgia, Chatham County, Georgia, Houston County, Georgia, Hall County, Georgia

---

### Q04 — L1 simple lookup

**Query:** How many people live in the city of Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **9/10** — 4 plan calls; 4 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 7.7s · 4 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000035_091899`

**Response prose:**

> In 2024, the city of Atlanta had an estimated population of 505,268 people. This represents a flat trend, up 2.7% from 492,204 people in 2021. For comparison, the state of Georgia had a population of 11,180,878, and the United States had a population of 334,922,499 in the same year.

**Key findings:**
- Atlanta city's population was 505,268 in 2024.
- The population increased by 2.7% from 2021 to 2024.
- Try next: How has the population changed here recently?
- Try next: What is the age breakdown of the population here?
- Try next: How does the population here compare to the surrounding metro area?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B01003`
- `acs/acs5/2021/B01003`
- `acs/acs5/2024/B01003`

**Peer comparisons:**
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q05 — L1 simple lookup

**Query:** What percentage of households in Gwinnett County are renters?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 24.6s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000052_838797`

**Response prose:**

> In 2024, 15.3% of households in Gwinnett County were renters. This rate is roughly in line with the Atlanta-Sandy Springs-Alpharetta, GA Metro Area (14.9%) and the state of Georgia (15.3%), and slightly higher than the national rate of 14.1%. The percentage of renter households in Gwinnett County has been falling, down 10.5% from 17.1% in 2021.

**Key findings:**
- In 2024, 15.3% of households in Gwinnett County were renters.
- This rate is comparable to the metro area and state, and slightly higher than the national average.
- The renter household percentage in Gwinnett County has decreased by 10.5% since 2021.
- Try next: What share of renters here are cost-burdened?
- Try next: How has the renter rate changed in this area recently?
- Try next: What is the median rent in this area?

**Citations** (2 rows, 1 unique tables):
- `acs/acs5/subject/2021/S2501`
- `acs/acs5/subject/2024/S2501`

**Peer comparisons:**
- *housing* (within state 13): Cobb County, Georgia, Henry County, Georgia, Cherokee County, Georgia, DeKalb County, Georgia, Coweta County, Georgia

---

### Q06 — L2 variable domain

**Query:** What's the median gross rent in Cobb County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 10.6s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000127_410887`

**Response prose:**

> In 2024, the median gross rent in Cobb County was $1,730. This figure is roughly in line with the Atlanta-Sandy Springs-Alpharetta, GA Metro Area's median gross rent of $1,770. Cobb County's median gross rent is higher than the state median of $1,506 and the national median of $1,413. Notably, the median gross rent in Cobb County has seen a significant increase, rising by 26.6% from $1,367 in 2021.

**Key findings:**
- In 2024, Cobb County's median gross rent was $1,730.
- This is comparable to the Atlanta-Sandy Springs-Alpharetta, GA Metro Area ($1,770).
- Cobb County's median gross rent is higher than both the state ($1,506) and national ($1,413) medians.
- Median gross rent in Cobb County increased by 26.6% since 2021.
- Try next: What share of households here are rent-burdened?
- Try next: How has median gross rent changed here recently?
- Try next: How does median gross rent here compare to the surrounding region?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B25064`
- `acs/acs5/2021/B25064`
- `acs/acs5/2024/B25064`

**Peer comparisons:**
- *housing* (within state 13): Gwinnett County, Georgia, Cherokee County, Georgia, DeKalb County, Georgia, Henry County, Georgia, Coweta County, Georgia

---

### Q07 — L2 variable domain

**Query:** What's the homeownership rate in Fulton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 25.1s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000148_029302`

**Response prose:**

> In 2024, Fulton County's homeownership rate was 54.3%. This rate is below the Atlanta-Sandy Springs-Alpharetta, GA Metro Area's rate of 66.8%, the state rate of 66.3%, and the national rate of 65.2%. The county's homeownership rate has seen a slight increase of 2.3% since 2021. The median home value in Fulton County was $458,800 in 2024, which is roughly in line with the metro level ($402,100) but notably higher than the state ($343,300) and national ($332,700) medians. This represents a significant increase of 32.9% since 2021. Meanwhile, the median household income in Fulton County was $95,292, also roughly in line with the metro area ($92,344) and up 22.7% from 2021. The combination of a lower homeownership rate and rapidly increasing home values, despite rising incomes, suggests potential affordability challenges contributing to a homeownership gap in the county.

**Key findings:**
- Fulton County's homeownership rate was 54.3% in 2024, which is below the metro, state, and national averages.
- Median home values in Fulton County rose significantly by 32.9% since 2021, reaching $458,800 in 2024.
- Median household income also increased by 22.7% since 2021, reaching $95,292 in 2024.
- The disparity between homeownership rates and rapidly increasing home values points to potential affordability barriers.
- Try next: How does homeownership vary by race or ethnicity here?
- Try next: What income is needed to afford a median-priced home here?
- Try next: How has the homeownership rate changed recently?

**Caveats:**
- ACS homeownership is by householder race; gaps at small-area level have wide margins of error.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25077`
- `acs/acs5/2021/B02001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25077`
- `acs/acs5/2021/B25081`
- `acs/acs5/2024/B02001`
- `acs/acs5/2024/B19013`
- `acs/acs5/2024/B25003`
- `acs/acs5/2024/B25077`
- …and 1 more

**Peer comparisons:**
- *housing* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia

---

### Q08 — L2 variable domain

**Query:** What's the uninsured rate in DeKalb County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 29.8s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000223_177446`

**Response prose:**

> In 2024, DeKalb County had an uninsured rate of 12.7%. This rate is notably higher than the national average, standing at 1.51 times the U.S. rate of 8.4%. While it is roughly in line with the Atlanta metro area's rate of 11.5% and Georgia's state rate of 12.0%, it remains elevated compared to the country as a whole. The uninsured rate in DeKalb County has seen a mild decrease since 2021, when it was 13.4%.
> >
> > Economically, DeKalb County's poverty rate in 2024 was 13.7%, which is somewhat above the metro average of 10.0% and slightly higher than the state (12.6%) and national (12.5%) rates. The median household income was $80,644, lower than the metro area's $92,344 but comparable to the state's $79,991 and the national $80,734. This income figure represents a strong increase of 16.2% since 2021.

**Key findings:**
- DeKalb County's uninsured rate was 12.7% in 2024, which is 1.51 times the national average.
- The county's uninsured rate has mildly decreased by 4.8% since 2021.
- The poverty rate in DeKalb County (13.7%) is somewhat above the metro average (10.0%).
- Median household income in DeKalb County ($80,644) is lower than the metro area but has seen a strong increase since 2021.
- Try next: How has the uninsured rate changed here recently?
- Try next: What's the uninsured rate for households below the poverty line?
- Try next: Are there racial or ethnic disparities in health coverage here?

**Caveats:**
- ACS distinguishes coverage type but not continuity; gaps during the year are under-measured.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B27010`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B27002`
- `acs/acs5/2021/B27003`
- `acs/acs5/2021/B27010`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- `acs/acs5/2024/B27002`
- `acs/acs5/2024/B27003`
- …and 1 more

**Peer comparisons:**
- *health_insurance* (within state 13): Chatham County, Georgia, Bibb County, Georgia, Richmond County, Georgia, Cobb County, Georgia, Houston County, Georgia
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia

---

### Q09 — L2 variable domain

**Query:** How many veterans live in Fulton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **9/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited
- **Runtime:** 12.4s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000302_931349`

**Response prose:**

> In 2024, Fulton County had approximately 38,720 veterans, representing 4.6% of its population. This percentage is notably low, falling below the metro level (0.75 times) and significantly below the state average (0.65 times), which is considered far below the expected range. The veteran population in Fulton County has also seen a mild decline, decreasing by 9.3% since 2021, when it was 5.0% of the population.

**Key findings:**
- Approximately 38,720 veterans lived in Fulton County in 2024, making up 4.6% of the population.
- The veteran population in Fulton County is lower than the metro, state, and national averages.
- The percentage of veterans in Fulton County is 0.65 times the state average, indicating it is far below the expected range.
- From 2021 to 2024, the veteran population in Fulton County decreased by 9.3%.
- Try next: What share of veterans here live in poverty?
- Try next: How many veterans here have a service-connected disability?
- Try next: How has the veteran population changed here recently?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B21001`
- `acs/acs5/2021/B21001`
- `acs/acs5/2024/B21001`

---

### Q10 — L2 variable domain

**Query:** What's the educational attainment in Clayton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 11.2s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000325_306089`

**Response prose:**

> In 2024, the educational attainment in Clayton County was 189,995. This figure is part of a rising trend, up 3.5% from 183,644 in 2021. For comparison, the Atlanta-Sandy Springs-Alpharetta, GA Metro Area had an educational attainment of 4,213,079, while Georgia's total was 7,359,821, and the United States had 230,807,303.

**Key findings:**
- Clayton County's educational attainment was 189,995 in 2024.
- This represents a 3.5% increase from 2021.
- Try next: How has educational attainment changed here recently?
- Try next: How does educational attainment vary by race or ethnicity?

**Citations** (2 rows, 1 unique tables):
- `acs/acs5/2021/B15003`
- `acs/acs5/2024/B15003`

**Peer comparisons:**
- *education* (within state 13): Richmond County, Georgia, Muscogee County, Georgia, Bibb County, Georgia, Hall County, Georgia, Paulding County, Georgia

---

### Q11 — L3 neighborhood

**Query:** What's the median income in Buckhead?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 6 plan calls; 6 aggregated values; 1 unique tables cited; 1 peer axis; 1 caveat
- **Runtime:** 9.0s · 6 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000346_520132`

**Response prose:**

> In 2024, the median income in Buckhead was $139,871. This figure is substantially above the metro level, and it represents a 6.8% increase from $130,943 in 2021. For comparison, the median income for Fulton County was $95,292, for the Atlanta-Sandy Springs-Alpharetta, GA Metro Area it was $92,344, for Georgia it was $79,991, and for the United States it was $80,734.

**Key findings:**
- Buckhead's median income of $139,871 in 2024 is significantly higher than county, metro, state, and national averages.
- The median income in Buckhead has risen by 6.8% since 2021.
- Try next: How has median income changed here recently?
- Try next: How does this median income compare to the broader metro area?
- Try next: What is the income distribution by household type or race?

**Caveats:**
- The median income value for Buckhead is an unweighted mean across 33 tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B19013`
- `acs/acs5/2021/B19013`
- `acs/acs5/2024/B19013`

**Peer comparisons:**
- *economic* (nationwide size-matched): Grant Park, Kirkwood, Reynoldstown, Old Fourth Ward, Home Park

---

### Q12 — L3 neighborhood

**Query:** What's the poverty rate in Midtown Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **9/10** — 6 plan calls; 6 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 9.5s · 6 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000405_498538`

**Response prose:**

> In 2024, the poverty rate in Midtown was 12.3%. This rate is somewhat above the Atlanta-Sandy Springs-Alpharetta, GA Metro Area level of 10.0%, and it has risen by 12.5% since 2021. For comparison, the poverty rate in Fulton County was 12.3%, in Georgia it was 12.6%, and the national rate was 12.5%.

**Key findings:**
- Midtown's poverty rate in 2024 was 12.3%.
- This rate is somewhat above the metro area's rate (10.0%).
- The poverty rate in Midtown has increased by 12.5% since 2021.
- Try next: How has the poverty rate changed here recently?
- Try next: What is the poverty rate for children in this area?
- Try next: How does the poverty rate here compare to the city average?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2024/B17001`

**Peer comparisons:**
- *economic* (nationwide size-matched): North Buckhead, Virginia Highland, Old Fourth Ward, Morningside/Lenox Park, Grant Park

---

### Q13 — L3 neighborhood

**Query:** How many households in East Atlanta have no vehicle?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 40 plan calls; 30 aggregated values; 5 unique tables cited; 2 peer axises; 2 caveats
- **Runtime:** 27.5s · 40 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000424_987380`

**Response prose:**

> In 2024, approximately 23,899 households in East Atlanta, representing 12.3% of all households, had no vehicle available. This rate is notably high, standing at 2.30 times the metro average, indicating a significant transportation challenge for residents. This figure has seen a mild decrease of 5.6% since 2021.
> >
> > The area also faces economic pressures, with a poverty rate of 14.3% in 2024, which is substantially above the metro level. The median household income in East Atlanta was $103,181 in 2024, roughly in line with the metro area, and has shown a strong increase of 20.9% since 2021. The combination of a high rate of carless households and elevated poverty suggests significant barriers to accessing essential services and employment opportunities.

**Key findings:**
- In 2024, 12.3% of households in East Atlanta (approximately 23,899 households) had no vehicle available.
- This rate is 2.30 times higher than the metro average, indicating a significant transportation challenge.
- The share of carless households decreased by 5.6% since 2021.
- East Atlanta's poverty rate was 14.3% in 2024, substantially above the metro level.
- The median household income was $103,181 in 2024, showing a strong increase since 2021.
- Try next: How has the share of households without a vehicle changed recently?
- Try next: Are carless households disproportionately concentrated among certain racial or ethnic groups?
- Try next: What share of carless households are also in a food desert?

**Caveats:**
- Household-level vehicle data may understate individual access constraints within multi-adult households.
- The median household income for East Atlanta is an unweighted mean across tracts; true sub-county medians would require population-weighted interpolation.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25044`
- `acs/acs5/2021/B08006`
- `acs/acs5/2021/B08203`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25044`
- `acs/acs5/2024/B08006`
- `acs/acs5/2024/B08203`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- …and 1 more

**Peer comparisons:**
- *commute_transit* (nationwide size-matched): Minneapolis city, Minnesota, Portland city, Oregon, Denver city, Colorado, Alexandria city, Virginia, Miami city, Florida
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q14 — L3 neighborhood

**Query:** What's the rent burden rate in Old Fourth Ward?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 30 plan calls; 30 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 16.5s · 30 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000502_525206`

**Response prose:**

> In 2024, Old Fourth Ward had a rent burden rate of 31.8%, which is far below the Atlanta-Sandy Springs-Alpharetta, GA Metro Area rate of 53.2%. However, this rate represents a significant increase of 54.2% since 2021, indicating a notable shift in housing affordability. The median gross rent in Old Fourth Ward was $2,290 in 2024, which is elevated and far above the national average of $1,413. This median gross rent has also seen a substantial increase of 100.0% since 2021.
> >
> > Despite a median household income of $104,271 in 2024, which is roughly in line with the metro level and up 78.2% since 2021, the neighborhood faces significant housing instability challenges. The homeownership rate is very low at 26.0%, far below the metro level of 66.8%. Furthermore, the poverty rate in Old Fourth Ward is very high at 34.1%, which is far above the metro level of 10.0% and has risen by 22.4% since 2021. These factors suggest that while the overall rent burden rate appears lower than comparators, the rapid increase in rent and the high poverty rate point to underlying housing instability.

**Key findings:**
- The rent burden rate in Old Fourth Ward was 31.8% in 2024, far below the metro average of 53.2%.
- The rent burden rate increased by 54.2% since 2021.
- Median gross rent in Old Fourth Ward was $2,290 in 2024, which is elevated and 1.62 times the national average, and has doubled since 2021.
- The homeownership rate is very low at 26.0%, significantly lower than the metro rate of 66.8%.
- The poverty rate is very high at 34.1%, more than three times the metro average of 10.0%.
- Try next: What is the rent burden for households at different income levels here?
- Try next: How has the racial composition of the neighborhood changed recently?
- Try next: What are the recent eviction rates in this area?

**Caveats:**
- Rent and cost-burden data reflect 5-year ACS averages; they lag current conditions by up to 3 years.

**Citations** (15 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25064`
- `acs/acs1/2024/B25070`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25064`
- `acs/acs5/2021/B25070`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- …and 3 more

**Peer comparisons:**
- *housing* (nationwide size-matched): Brookwood, Collier Hills North, Sweet Auburn, Lindridge/Martin Manor, Underwood Hills
- *economic* (nationwide size-matched): Pine Hills, Virginia Highland, Grant Park, Morningside/Lenox Park, Kirkwood

---

### Q15 — L3 neighborhood

**Query:** What's the child poverty rate in the Westside of Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 40 plan calls; 30 aggregated values; 5 unique tables cited; 3 peer axises; 2 caveats
- **Runtime:** 15.8s · 40 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000528_985417`

**Response prose:**

> In 2024, the Westside of Atlanta faces significant challenges in child welfare, with a child poverty rate of 6.5%. This rate is notably high, nearly double the Fulton County rate (3.4%) and the metro Atlanta rate (3.5%), and 1.8 times the national average. While still elevated, this represents a strong decrease of 13.4% from 2021.
> >
> > Contributing to these challenges, 24.2% of households in the Westside of Atlanta are single-parent households, which is substantially higher than the metro level (13.6%) and 1.86 times the national average. The child uninsured rate stands at 13.5%, which is somewhat above the metro level (11.5%) and 1.6 times the national average, though it has fallen by 9.9% since 2021. The median household income in the Westside of Atlanta is $77,850, which is below the metro level ($92,344) but has seen a strong increase of 24.4% from 2021.

**Key findings:**
- The child poverty rate in the Westside of Atlanta was 6.5% in 2024, which is nearly double the county and metro rates, and 1.8 times the national average.
- This child poverty rate has shown a strong decrease of 13.4% since 2021.
- Single-parent households constitute 24.2% of households, substantially higher than metro and national averages.
- The child uninsured rate is 13.5%, elevated compared to metro and national figures, but has decreased by 9.9% since 2021.
- Median household income is $77,850, below the metro average, but has risen by 24.4% since 2021.
- Try next: What share of families here receive SNAP benefits?
- Try next: How many households with children are rent-burdened?
- Try next: What share of children here lack broadband internet access?

**Caveats:**
- The value for 'share under 18' is the unweighted mean across 55 tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.
- The value for 'median household income' is the unweighted mean across 52 tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B09001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B27010`
- `acs/acs5/2021/B09001`
- `acs/acs5/2021/B11003`
- `acs/acs5/2021/B17020`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B27010`
- `acs/acs5/2024/B09001`
- `acs/acs5/2024/B11003`
- `acs/acs5/2024/B17020`
- `acs/acs5/2024/B19013`
- …and 1 more

**Peer comparisons:**
- *age_structure* (nationwide size-matched): Boston city, Massachusetts, Salt Lake City city, Utah, Minneapolis city, Minnesota, Richmond city, Virginia, Washington city, District of Columbia
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia
- *housing* (nationwide size-matched): Denver city, Colorado, Orlando city, Florida, Austin city, Texas, Raleigh city, North Carolina, Tampa city, Florida

---

### Q16 — L4 grant frame

**Query:** I'm writing a food pantry grant for Southwest Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 30 plan calls; 30 aggregated values; 5 unique tables cited; 3 peer axises; 2 caveats
- **Runtime:** 31.0s · 30 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000554_799214`

**Response prose:**

> In 2024, the southwest side of Atlanta city demonstrates a significant need for food assistance, characterized by high rates of SNAP participation, poverty, and limited transportation access. The SNAP participation rate stands at 27.5%, which is far above the metro level (3.38 times) and indicates a substantial reliance on food assistance programs. The area's poverty rate is 22.3%, far exceeding the metro average (2.22 times), though it has seen a mild decrease of 3.6% since 2021. Child poverty is also notably high at 6.8%, more than double the county average (2.02 times), despite a strong 16.4% decrease since 2021. The median household income in the area is $57,568, which is below the metro level (0.62 times), further underscoring the economic vulnerability, though it has seen a strong 24.3% increase since 2021. A critical barrier to food access is transportation: 18.6% of households in southwest Atlanta city do not have a vehicle, which is far above the metro level (3.47 times). This lack of transportation, while showing a strong 10.9% decrease since 2021, severely limits residents' ability to reach grocery stores and healthy food options.

**Key findings:**
- In 2024, 27.5% of residents in southwest Atlanta city participated in SNAP, 3.38 times the metro rate.
- The poverty rate was 22.3% in 2024, 2.22 times the metro rate, with child poverty at 6.8%, over twice the county rate.
- 18.6% of households lacked a vehicle in 2024, 3.47 times the metro rate, posing a significant barrier to food access.
- Median household income was $57,568 in 2024, 0.62 times the metro level.
- Try next: Is this area considered a food desert, or how far is the nearest grocery store?
- Try next: What is the rate of child food insecurity in this area?
- Try next: How have poverty rates or SNAP participation changed here recently?

**Caveats:**
- The median household income value is an unweighted mean across 31 tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.
- Census data approximates food-access conditions; direct food-insecurity measures come from USDA ERS / Feeding America.

**Citations** (14 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B22003`
- `acs/acs1/2024/B25044`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B17020`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B22003`
- `acs/acs5/2021/B25044`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B17020`
- `acs/acs5/2024/B19013`
- …and 2 more

**Peer comparisons:**
- *food_benefits* (nationwide size-matched): Columbus city, Ohio, Kansas City city, Kansas, Killeen city, Texas, Columbus city, Georgia, Chattanooga city, Tennessee
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia
- *commute_transit* (nationwide size-matched): Minneapolis city, Minnesota, Portland city, Oregon, Denver city, Colorado, Alexandria city, Virginia, Miami city, Florida

---

### Q17 — L4 grant frame

**Query:** Help me build a rent burden case for Fulton County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 17.3s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000635_785417`

**Response prose:**

> In 2024, Fulton County faces significant housing instability challenges, with nearly half (48.5%) of its renters experiencing rent burden. This rate is roughly in line with state and national levels but has seen a mild increase of 3.5% since 2021. A key driver of this burden is the rapidly increasing cost of housing; median gross rent in Fulton County was $1,732 in 2024, marking a strong 26.7% increase since 2021—a notable shift. While median household income also rose strongly by 22.7% to $95,292 over the same period, the faster pace of rent increases suggests a widening affordability gap. The county's homeownership rate of 54.3% in 2024 is low compared to the wider metro area, state, and national averages, indicating a larger proportion of residents who are renters and thus more susceptible to rent burden. Furthermore, Fulton County's poverty rate of 12.3% in 2024 is elevated compared to the metro area, underscoring the vulnerability of a significant portion of its population to housing insecurity.

**Key findings:**
- Nearly half (48.5%) of Fulton County renters were rent-burdened in 2024, a rate that has mildly risen since 2021.
- Median gross rent in Fulton County increased by a strong 26.7% to $1,732 between 2021 and 2024, outpacing the 22.7% rise in median household income.
- The homeownership rate in Fulton County (54.3% in 2024) is low compared to metro, state, and national averages.
- Fulton County's poverty rate (12.3% in 2024) is elevated compared to the metro area.
- Try next: What share of low-income households here are rent-burdened?
- Try next: How has the racial composition of renters changed in this area recently?
- Try next: How has the overall rent burden changed here over the past few years?

**Caveats:**
- Rent and cost-burden data reflect 5-year ACS averages; they lag current conditions by up to 3 years.

**Citations** (15 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25064`
- `acs/acs1/2024/B25070`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25064`
- `acs/acs5/2021/B25070`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- …and 3 more

**Peer comparisons:**
- *housing* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia

---

### Q18 — L4 grant frame

**Query:** I need an ESL services grant narrative for Clarkston, Georgia.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **5/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 4.3s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000635_785417`

**Response prose:**

> I cannot generate a specific ESL services grant narrative for Clarkston, Georgia, as no data was provided to support such a narrative. To create an effective grant narrative, I would need specific demographic, economic, or educational data points relevant to ESL services in Clarkston.

---

### Q19 — L4 grant frame

**Query:** Workforce development grant for DeKalb County — what data should I cite?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 40.4s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000717_376151`

**Response prose:**

> DeKalb County faces significant workforce challenges, as indicated by 2024 data. The unemployment rate stands at 6.39%, which is substantially above the Atlanta metro area's rate of 4.55% and the state rate of 4.69%. While this rate has seen a mild decrease of 4.5% since 2021, it remains elevated. The county's poverty rate is 13.69%, somewhat above the metro area's 10.04% and slightly higher than the state and national averages.
> >
> > Despite these challenges, DeKalb County demonstrates strengths in educational attainment, with 47.43% of residents holding a bachelor's degree or higher in 2024, roughly in line with the metro area's 43.85% and notably higher than the state (36.31%) and national (35.68%) averages. Median household income in 2024 was $80,644, which is below the metro area's $92,344 but comparable to the state and national figures. This income has seen a strong increase of 16.2% since 2021.
> >
> > However, the share of youth not in school or employed in DeKalb County was 61.5% in 2024, which is roughly in line with the metro area's 64.4% but represents a strong increase of 10.8% since 2021, indicating a growing concern for youth engagement in the workforce.

**Key findings:**
- In 2024, DeKalb County's unemployment rate was 6.39%, substantially higher than the Atlanta metro area (4.55%) and the state (4.69%), though it has mildly decreased by 4.5% since 2021.
- The poverty rate in DeKalb County was 13.69% in 2024, which is elevated compared to the metro area (10.04%).
- Educational attainment is a strength, with 47.43% of residents holding a bachelor's degree or higher in 2024, exceeding state and national averages.
- Median household income in DeKalb County was $80,644 in 2024, showing a strong increase of 16.2% since 2021.
- The share of youth not in school or employed was 61.5% in 2024, a strong increase of 10.8% since 2021, highlighting a growing concern for youth workforce engagement.
- Try next: What are the most in-demand skills or industries here?
- Try next: How do unemployment rates differ by race or age group?
- Try next: How has the unemployment rate changed here recently?

**Caveats:**
- ACS employment data is self-reported and reflects the 5-year survey window.

**Citations** (14 rows, 5 unique tables):
- `acs/acs1/2024/B15003`
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B23025`
- `acs/acs1/subject/2021/S2301`
- `acs/acs1/subject/2024/S2301`
- `acs/acs5/2021/B15003`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B23025`
- `acs/acs5/2024/B15003`
- `acs/acs5/2024/B17001`
- …and 2 more

**Peer comparisons:**
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia
- *education* (within state 13): Cobb County, Georgia, Forsyth County, Georgia, Gwinnett County, Georgia, Chatham County, Georgia, Columbia County, Georgia
- *age_structure* (within state 13): Muscogee County, Georgia, Cobb County, Georgia, Houston County, Georgia, Columbia County, Georgia, Paulding County, Georgia

---

### Q20 — L4 grant frame

**Query:** Building a case for senior isolation programs in Cobb County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 29 plan calls; 29 aggregated values; 6 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 98.6s · 29 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000807_785252`

**Response prose:**

> Cobb County presents a compelling case for senior isolation programs, with 174,777 seniors living alone in 2024. While this figure is proportionally lower than the broader Atlanta metro area, this substantial absolute number highlights a significant population vulnerable to social isolation. The number of seniors living alone has remained relatively stable since 2021. Financial hardship can exacerbate isolation; in 2024, 8.7% of seniors in Cobb County lived in poverty, a rate comparable to the metro area and showing a slight increase of 1.5% since 2021. Transportation access is another critical factor: 3.9% of Cobb County households lacked a vehicle in 2024. Although this rate is significantly lower than national averages, it has seen a strong increase of 11.9% since 2021, indicating a growing barrier for seniors to access community resources and maintain social connections. On a positive note, Cobb County boasts a high broadband subscription rate of 96.2% in 2024, comparable to the metro area, which has also seen a slight increase of 1.5% since 2021. This robust connectivity offers a strong foundation for implementing and expanding remote services to combat senior isolation. In 2020, 60.5% of senior isolation programs were active in Cobb County, a figure similar to the metro area, suggesting a baseline of existing support that could be enhanced to meet the identified needs.

**Key findings:**
- 174,777 seniors lived alone in Cobb County in 2024, representing a substantial population vulnerable to isolation.
- 8.7% of seniors in Cobb County lived in poverty in 2024, a rate comparable to the metro area and slightly increasing since 2021.
- 3.9% of Cobb County households lacked a vehicle in 2024, a rate that has strongly increased by 11.9% since 2021.
- Cobb County has a high broadband subscription rate of 96.2% in 2024, offering strong potential for remote senior support services.

**Caveats:**
- Isolation is inferred from household composition and access indicators; direct measures of social isolation are not in ACS.

**Citations** (14 rows, 6 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B25044`
- `acs/acs1/2024/B28002`
- `acs/acs5/2021/B01001`
- `acs/acs5/2021/B09010`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B25044`
- `acs/acs5/2021/B28002`
- `acs/acs5/2024/B01001`
- `acs/acs5/2024/B09010`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B25044`
- …and 2 more

**Peer comparisons:**
- *age_structure* (within state 13): Houston County, Georgia, Columbia County, Georgia, Paulding County, Georgia, Henry County, Georgia, DeKalb County, Georgia
- *economic* (within state 13): Gwinnett County, Georgia, DeKalb County, Georgia, Cherokee County, Georgia, Forsyth County, Georgia, Columbia County, Georgia
- *commute_transit* (within state 13): Gwinnett County, Georgia, Cherokee County, Georgia, Forsyth County, Georgia, Henry County, Georgia, DeKalb County, Georgia

---

### Q21 — L5 compare/trend

**Query:** Compare poverty rates between DeKalb and Fulton counties.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 5 plan calls; 5 aggregated values; 1 unique tables cited; 1 peer axis; 1 caveat
- **Runtime:** 17.3s · 5 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_000956_365682`

**Response prose:**

> Data for Fulton County's poverty rate was not available for comparison. In 2024, DeKalb County had a poverty rate of 13.7%. This rate is somewhat above the Atlanta-Sandy Springs-Alpharetta metro area's rate of 10.0%, and also slightly higher than both the Georgia state rate of 12.6% and the national rate of 12.5%. Compared to 2021, DeKalb County's poverty rate has remained relatively flat, increasing by 1.4% from 13.5%.

**Key findings:**
- In 2024, DeKalb County's poverty rate was 13.7%.
- This rate was elevated compared to the Atlanta metro area (10.0%), Georgia (12.6%), and the US (12.5%).
- DeKalb County's poverty rate in 2024 was relatively flat compared to 2021 (13.5%).
- Try next: How has the poverty rate changed in these counties recently?
- Try next: What is the child poverty rate in these counties?
- Try next: How do poverty rates differ by race or ethnicity in these areas?

**Caveats:**
- Data for Fulton County was not available in the provided dataset, preventing a direct comparison.

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2024/B17001`

**Peer comparisons:**
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia

---

### Q22 — L5 compare/trend

**Query:** How has median household income changed in Atlanta since 2019?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 14 plan calls; 14 aggregated values; 1 unique tables cited; 1 peer axis; 1 caveat
- **Runtime:** 16.4s · 14 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001023_667715`

**Response prose:**

> In Atlanta city, the median household income has shown a consistent increase since 2020. In 2020, it was $64,179, which was below the state level and the national average. By 2024, the median household income in Atlanta city rose to $85,652. This 2024 figure is roughly in line with the state level of $79,991 (1.07x) and the national average of $80,734 (1.06x). Data for 2019 was not available.

**Key findings:**
- Atlanta city's median household income increased from $64,179 in 2020 to $85,652 in 2024.
- In 2020, Atlanta's median household income was below both the state and national averages.
- By 2024, Atlanta's median household income was roughly in line with both the state and national averages.
- Try next: What does the income distribution look like here?
- Try next: How does median income vary by race or ethnicity here?
- Try next: How does the median income here compare to the cost of living?

**Caveats:**
- Data for median household income in Atlanta city for the year 2019 was not available.

**Citations** (9 rows, 1 unique tables):
- `acs/acs1/2021/B19013`
- `acs/acs1/2022/B19013`
- `acs/acs1/2023/B19013`
- `acs/acs1/2024/B19013`
- `acs/acs5/2020/B19013`
- `acs/acs5/2021/B19013`
- `acs/acs5/2022/B19013`
- `acs/acs5/2023/B19013`
- `acs/acs5/2024/B19013`

**Peer comparisons:**
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q23 — L5 compare/trend

**Query:** How has SNAP participation changed in Clayton County over the last 5 years?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 110 plan calls; 110 aggregated values; 5 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 23.3s · 110 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001050_036772`

**Response prose:**

> In Clayton County, SNAP participation has remained relatively stable over the last five years, fluctuating between 18.7% and 20.0%. In 2020, 20.0% of the population participated in SNAP, a rate far above the metro level (2.46 times). By 2024, the rate was 19.3%, which was still far above the metro level (2.37 times).
> >
> > This consistently high SNAP participation aligns with other indicators of economic hardship in the county. The poverty rate in Clayton County was 17.9% in 2024, substantially above the metro level (1.79 times). Similarly, the child poverty rate was 7.3% in 2024, far above the metro level (2.09 times). The median household income in 2024 was $59,806, which was below the metro level (0.65 times).
> >
> > Furthermore, access to transportation, which is crucial for food access, is a notable concern. In 2024, 7.1% of households in Clayton County were without a vehicle, a figure somewhat above the metro level (1.32 times).

**Key findings:**
- SNAP participation in Clayton County has been consistently high, ranging from 18.7% to 20.0% between 2020 and 2024.
- The county's SNAP participation rate is significantly higher than the metro, state, and national averages, being over 2.3 times the metro level in 2024.
- Poverty and child poverty rates are also substantially higher than metro averages, with the 2024 poverty rate at 17.9% and child poverty rate at 7.3%.
- Median household income in Clayton County ($59,806 in 2024) is considerably lower than the metro average.
- A notable share of households (7.1% in 2024) lack a vehicle, indicating potential barriers to food access.
- Try next: Is this area considered a food desert?
- Try next: What share of SNAP households here have children?
- Try next: What is the food insecurity rate in this area?

**Caveats:**
- Census data approximates food-access conditions; direct food-insecurity measures come from USDA ERS / Feeding America.

**Citations** (41 rows, 5 unique tables):
- `acs/acs1/2021/B17001`
- `acs/acs1/2021/B19013`
- `acs/acs1/2021/B22003`
- `acs/acs1/2021/B25044`
- `acs/acs1/2022/B17001`
- `acs/acs1/2022/B19013`
- `acs/acs1/2022/B22003`
- `acs/acs1/2022/B25044`
- `acs/acs1/2023/B17001`
- `acs/acs1/2023/B19013`
- `acs/acs1/2023/B22003`
- `acs/acs1/2023/B25044`
- …and 29 more

**Peer comparisons:**
- *food_benefits* (within state 13): Muscogee County, Georgia, Richmond County, Georgia, DeKalb County, Georgia, Bibb County, Georgia, Chatham County, Georgia
- *economic* (within state 13): Richmond County, Georgia, Henry County, Georgia, Chatham County, Georgia, Houston County, Georgia, Hall County, Georgia
- *commute_transit* (within state 13): Hall County, Georgia, Chatham County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia, Henry County, Georgia

---

### Q24 — L5 compare/trend

**Query:** Compare rent burden between Atlanta city and the Atlanta MSA.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 20 plan calls; 20 aggregated values; 5 unique tables cited; 2 peer axises; 2 caveats
- **Runtime:** 25.2s · 20 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001123_353130`

**Response prose:**

> Data for the Atlanta MSA is not available in this dataset, so a direct comparison of rent burden between Atlanta city and the Atlanta MSA cannot be made. In 2024, Atlanta city experienced a rent burden of 49.0%, which is roughly in line with the state of Georgia (48.8%) and slightly above the national average of 47.6%. This represents a mild increase of 4.7% since 2021.
> >
> > Housing instability in Atlanta city is further indicated by its low homeownership rate of 46.4% in 2024, significantly below Georgia's 66.3% and the national 65.2%. Median gross rent in Atlanta city was $1,711 in 2024, which is notably higher than the state's $1,506 and the national $1,413. This figure has seen a strong increase of 27.5% since 2021. While median household income in Atlanta city also rose strongly by 23.8% to $85,652, its poverty rate of 16.9% remains elevated compared to the state (12.6%) and national (12.5%) averages, although it has seen a mild decrease since 2021.

**Key findings:**
- Atlanta city's rent burden was 49.0% in 2024, slightly higher than state and national averages, and has risen since 2021.
- The city's homeownership rate of 46.4% is substantially lower than state and national figures.
- Median gross rent in Atlanta city was $1,711 in 2024, significantly higher than state and national averages, and increased by 27.5% since 2021.
- Atlanta city's median household income was $85,652 in 2024, showing strong growth, but its poverty rate of 16.9% remains elevated.
- Try next: How has the racial composition of this area changed recently?
- Try next: What's the rent burden for different income levels here?
- Try next: What are the recent eviction rates in this area?

**Caveats:**
- Data for the Atlanta MSA was not available.
- Rent and cost-burden data reflect 5-year ACS averages; they lag current conditions by up to 3 years.

**Citations** (15 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25064`
- `acs/acs1/2024/B25070`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25064`
- `acs/acs5/2021/B25070`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- …and 3 more

**Peer comparisons:**
- *housing* (nationwide size-matched): Denver city, Colorado, Orlando city, Florida, Austin city, Texas, Raleigh city, North Carolina, Tampa city, Florida
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q25 — L5 compare/trend

**Query:** How has child poverty changed in Fulton County since 2019?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 94 plan calls; 94 aggregated values; 5 unique tables cited; 3 peer axises; 3 caveats
- **Runtime:** 27.6s · 94 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001158_514432`

**Response prose:**

> In Fulton County, the child poverty rate has generally declined since 2020. It was 4.3% in 2020, which was somewhat above the metro level. By 2024, the rate fell to 3.4%, roughly in line with the metro level of 3.5%, below the state rate of 4.3%, and similar to the national rate of 3.6%. This positive trend is supported by an increase in median household income, which rose from $72,741 in 2020 to $95,292 in 2024, reaching a level roughly in line with the metro area. The proportion of single-parent households, a factor in child welfare, also decreased from 16.7% in 2020 to 15.7% in 2024, though it remains somewhat above the metro level. The child uninsured rate declined from 10.1% in 2020 to 9.5% in 2024, which is now below the metro level.

**Key findings:**
- Child poverty in Fulton County decreased from 4.3% in 2020 to 3.4% in 2024.
- The 2024 child poverty rate of 3.4% is roughly in line with the metro level and below the state average.
- Median household income in Fulton County increased from $72,741 in 2020 to $95,292 in 2024.
- The proportion of single-parent households decreased from 16.7% in 2020 to 15.7% in 2024, remaining somewhat above the metro level.
- The child uninsured rate declined from 10.1% in 2020 to 9.5% in 2024, which is below the metro level.
- Try next: What is the child food insecurity rate here?
- Try next: What share of children in poverty live in single-parent households?
- Try next: How does the child poverty rate here compare to the state average?

**Caveats:**
- Data for child poverty in Fulton County is available starting from 2020, not 2019.
- Small-area child-specific estimates have wide confidence intervals.
- The share of the population under 18 in Fulton County is far below the metro average.

**Citations** (37 rows, 5 unique tables):
- `acs/acs1/2021/B09001`
- `acs/acs1/2021/B19013`
- `acs/acs1/2021/B27010`
- `acs/acs1/2022/B09001`
- `acs/acs1/2022/B19013`
- `acs/acs1/2022/B27010`
- `acs/acs1/2023/B09001`
- `acs/acs1/2023/B19013`
- `acs/acs1/2023/B27010`
- `acs/acs1/2024/B09001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B27010`
- …and 25 more

**Peer comparisons:**
- *age_structure* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia
- *housing* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia

---

### Q26 — L6 disparity

**Query:** What's the racial income gap in Fulton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 29 plan calls; 29 aggregated values; 6 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 33.3s · 29 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001236_128165`

**Response prose:**

> In Fulton County, the aggregated racial income gap for 2021 was $54,690,093,100. This figure is far below the metro level (0.24 times) and the state level (0.15 times), reflecting the county's smaller economic scale compared to these larger geographies. More granularly, significant disparities exist in median household incomes by race in 2024. The median household income for White residents was $137,968, an elevated figure that is notably 1.60 times the national average and represents a strong increase of 22.2% from 2021. In contrast, Black residents had a median household income of $62,851, which is below the metro level (0.83 times) but still above state and national averages, showing a strong increase of 27.1% from 2021. Hispanic residents had a median household income of $86,348, roughly in line with the metro level (1.13 times), also experiencing a strong increase of 26.7% from 2021. The notable increases in median household income for Black and Hispanic residents since 2021 represent significant shifts.
> >
> > Regarding poverty in 2024, the poverty rate for Black residents in Fulton County was 19.2%, which is elevated and somewhat above the metro level (1.29 times). This rate has seen a mild decrease of 4.8% since 2021. The poverty rate for Hispanic residents was 13.9%, which is typical compared to the metro level (0.87 times) and has seen a strong decrease of 10.7% since 2021.

**Key findings:**
- White residents in Fulton County have a significantly higher median household income ($137,968 in 2024) compared to Black residents ($62,851) and Hispanic residents ($86,348).
- The median household income for White residents is notably 1.60 times the national average, while Black residents' median income is below the metro average.
- All racial groups (Black, White, Hispanic) in Fulton County experienced strong increases in median household income between 2021 and 2024.
- The poverty rate for Black residents in Fulton County (19.2% in 2024) is elevated and higher than the metro average.
- The poverty rate for Hispanic residents (13.9% in 2024) is typical and has seen a strong decrease since 2021.
- Try next: What are the educational attainment levels by race in this area?
- Try next: What is the homeownership rate by race here?
- Try next: What are the unemployment rates by race in this area?

**Caveats:**
- Income-by-race medians at sub-state levels carry wide margins of error; it's advisable to combine with longer 5-year windows for more robust analysis.

**Citations** (14 rows, 6 unique tables):
- `acs/acs1/2024/B19013A`
- `acs/acs1/2024/B19013B`
- `acs/acs1/2024/B19013I`
- `acs/acs5/2021/B17001B`
- `acs/acs5/2021/B17001I`
- `acs/acs5/2021/B19013A`
- `acs/acs5/2021/B19013B`
- `acs/acs5/2021/B19013I`
- `acs/acs5/2024/B17001B`
- `acs/acs5/2024/B17001I`
- `acs/acs5/2024/B19013A`
- `acs/acs5/2024/B19013B`
- …and 2 more

**Peer comparisons:**
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia
- *race_ethnicity* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia

---

### Q27 — L6 disparity

**Query:** Compare poverty rates by race in DeKalb County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 30 plan calls; 30 aggregated values; 6 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 18.4s · 30 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001319_457763`

**Response prose:**

> In 2024, DeKalb County experienced an overall poverty rate of 13.7%, which is somewhat above the Atlanta metro area's rate of 10.0% and the state rate of 12.6%. This rate has remained flat, increasing by 1.4% since 2021. However, poverty rates vary significantly by race, highlighting substantial disparities within the county. For Black individuals in DeKalb County, the poverty rate was 17.0% in 2024. This is roughly in line with the metro area's Black poverty rate of 14.8% but lower than the state (19.0%) and national (21.2%) rates. This rate has seen a mild rise of 3.5% since 2021. The median household income for Black individuals in DeKalb County was $63,329, which is below the metro level of $75,304, despite a strong 15.2% increase since 2021. Hispanic individuals in DeKalb County faced a poverty rate of 18.2% in 2024, which is roughly in line with the metro area (16.0%) but higher than the state (17.5%) and national (16.7%) rates. Notably, this rate has seen a strong decrease of 16.0% since 2021. In stark contrast to the Black median household income, White individuals in DeKalb County had a median household income of $129,386, which is elevated compared to the metro area's $107,625, and saw a strong increase of 22.1% since 2021, underscoring a significant racial income gap.

**Key findings:**
- In 2024, DeKalb County's overall poverty rate was 13.7%, elevated compared to the metro area (10.0%) and state (12.6%).
- The poverty rate for Black individuals in DeKalb County was 17.0% in 2024, higher than the metro area (14.8%) and showing a mild increase of 3.5% since 2021.
- Hispanic individuals in DeKalb County had a poverty rate of 18.2% in 2024, higher than metro, state, and national averages, though this rate decreased strongly by 16.0% since 2021.
- A significant income disparity exists: the median household income for Black individuals was $63,329, below the metro level, while for White individuals it was $129,386, elevated compared to the metro area.
- Try next: What are the child poverty rates by race here?
- Try next: What share of households by race are rent-burdened in this area?
- Try next: What is the SNAP participation rate by race here?

**Caveats:**
- Race-disaggregated ACS estimates have wider margins of error; consider combining with 5-year data for small geographies.

**Citations** (15 rows, 6 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013A`
- `acs/acs1/2024/B19013B`
- `acs/acs5/2021/B02001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B17001B`
- `acs/acs5/2021/B17001I`
- `acs/acs5/2021/B19013A`
- `acs/acs5/2021/B19013B`
- `acs/acs5/2024/B02001`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B17001B`
- …and 3 more

**Peer comparisons:**
- *race_ethnicity* (within state 13): Houston County, Georgia, Clayton County, Georgia, Bibb County, Georgia, Henry County, Georgia, Cobb County, Georgia
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia

---

### Q28 — L6 disparity

**Query:** Is there a gender earnings gap in Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 24 plan calls; 24 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 48.7s · 24 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001347_891560`

**Response prose:**

> In Atlanta in 2024, the reported gender earnings gap was $51,946. This figure is elevated, being somewhat above the state level ($39,250) and the national level ($40,685), and has risen by 22.1% since 2021. For context, female earnings by occupation in Atlanta were $75,824, which is substantially above the state level ($53,594) and has increased by 19.2% since 2021. Male earnings by occupation were $65,459, also elevated compared to the state ($47,066) and up 23.1% from 2021. Based on these occupational earnings, women in Atlanta earned $10,365 more than men in 2024. The overall labor force participation rate in Atlanta was 68.9%, roughly in line with the state level (64.5%) and up 3.6% from 2021.

**Key findings:**
- Atlanta's reported gender earnings gap was $51,946 in 2024, which is elevated compared to both state and national levels, and has increased by 22.1% since 2021.
- Female earnings by occupation in Atlanta were $75,824 in 2024, substantially higher than state and national averages, and up 19.2% from 2021.
- Male earnings by occupation in Atlanta were $65,459 in 2024, also elevated compared to state and national averages, and up 23.1% from 2021.
- In 2024, women in Atlanta earned $10,365 more than men in occupations.
- Atlanta's labor force participation rate was 68.9% in 2024, similar to the state average of 64.5%.
- Try next: How does the gender earnings gap differ across racial and ethnic groups here?
- Try next: What is the earnings gap for women with children compared to those without?
- Try next: How has the gender earnings gap changed in this area recently?

**Caveats:**
- The data on earnings by sex are reported for sex-binary individuals only and do not include non-binary or non-reported workers.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2021/B20002`
- `acs/acs1/2024/B20002`
- `acs/acs1/2024/B23025`
- `acs/acs5/2021/B20002`
- `acs/acs5/2021/B23001`
- `acs/acs5/2021/B23025`
- `acs/acs5/2021/B24031`
- `acs/acs5/2021/B24032`
- `acs/acs5/2024/B20002`
- `acs/acs5/2024/B23001`
- `acs/acs5/2024/B23025`
- `acs/acs5/2024/B24031`
- …and 1 more

**Peer comparisons:**
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia
- *employment* (nationwide size-matched): Jersey City city, New Jersey, San Francisco city, California, Raleigh city, North Carolina, Durham city, North Carolina, Plano city, Texas

---

### Q29 — L6 disparity

**Query:** Compare child poverty rates for Hispanic vs white children in Gwinnett.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 3 peer axises; 2 caveats
- **Runtime:** 31.9s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001446_543731`

**Response prose:**

> Data specifically comparing child poverty rates for Hispanic versus white children in Gwinnett County is not available in the provided dataset for 2024. Overall, Gwinnett County's child poverty rate in 2024 was 3.8%. This rate is roughly in line with the metro level (3.5%) and the national average (3.6%), while being lower than the state rate of 4.3%. The child poverty rate in Gwinnett County has remained flat, decreasing slightly by 2.4% from 3.9% in 2021.

**Key findings:**
- Data for child poverty rates specifically comparing Hispanic and white children in Gwinnett County is not available for 2024.
- In 2024, the overall child poverty rate in Gwinnett County was 3.8%.
- This rate is comparable to the metro (3.5%) and national (3.6%) averages, and lower than the state (4.3%) rate.
- The child poverty rate in Gwinnett County showed a flat trend, decreasing by 2.4% from 3.9% in 2021.
- Try next: What are the child poverty rates for other racial and ethnic groups here?
- Try next: How do child uninsured rates compare across racial and ethnic groups?
- Try next: What's the breakdown of single-parent households by race or ethnicity?

**Caveats:**
- The provided data does not include child poverty rates broken down by specific racial or ethnic groups (e.g., Hispanic vs. White children).
- Small-area child-specific estimates have wide confidence intervals; margin of error should be considered where material.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B09001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B27010`
- `acs/acs5/2021/B09001`
- `acs/acs5/2021/B11003`
- `acs/acs5/2021/B17020`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B27010`
- `acs/acs5/2024/B09001`
- `acs/acs5/2024/B11003`
- `acs/acs5/2024/B17020`
- `acs/acs5/2024/B19013`
- …and 1 more

**Peer comparisons:**
- *race_ethnicity* (within state 13): Cobb County, Georgia, Hall County, Georgia, Clayton County, Georgia, Muscogee County, Georgia, Chatham County, Georgia
- *age_structure* (within state 13): Paulding County, Georgia, Henry County, Georgia, Houston County, Georgia, Clayton County, Georgia, Cobb County, Georgia
- *economic* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Henry County, Georgia, Cherokee County, Georgia, Houston County, Georgia

---

### Q30 — L6 disparity

**Query:** What's the educational attainment gap by race in Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 24 plan calls; 24 aggregated values; 6 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 29.6s · 24 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001528_428740`

**Response prose:**

> In 2024, the value associated with the educational attainment gap by race in Atlanta was 355,043, reflecting a 4.8% increase from 2021. Beyond this, significant racial disparities are evident in income levels. The median household income for White residents in Atlanta was $134,739 in 2024, which is substantially above the state level and 1.56 times the national average. This figure saw a strong increase of 19.9% from 2021. In contrast, the median household income for Black residents was $51,197, which is below both the state and national averages. This income for Black residents, however, experienced a notable shift, rising strongly by 31.8% since 2021. Poverty rates also show disparities. The poverty rate for Black residents in Atlanta was 25.4% in 2024, which is elevated compared to both the state (1.34 times) and national (1.20 times) levels, though it decreased by 8.4% from 2021. For Hispanic residents, the poverty rate was 16.2%, roughly in line with state and national levels, and saw a strong decrease of 24.4% from 2021.

**Key findings:**
- The value for the educational attainment gap by race in Atlanta was 355,043 in 2024, up 4.8% from 2021.
- Median household income for White residents ($134,739) in Atlanta was significantly higher than for Black residents ($51,197) in 2024.
- White median household income was 1.56 times the national average, while Black median household income was below state and national averages.
- The poverty rate for Black residents (25.4%) was elevated compared to state and national averages in 2024.
- Try next: What are the unemployment rates for different racial groups here?
- Try next: What are the broadband subscription rates for different racial groups?
- Try next: What share of each racial group here has a bachelor's degree or higher?

**Caveats:**
- Race-disaggregated ACS estimates have wider MOEs; combine with 5-year data at small geographies.

**Citations** (14 rows, 6 unique tables):
- `acs/acs1/2024/B19013A`
- `acs/acs1/2024/B19013B`
- `acs/acs5/2021/B02001`
- `acs/acs5/2021/B15003`
- `acs/acs5/2021/B17001B`
- `acs/acs5/2021/B17001I`
- `acs/acs5/2021/B19013A`
- `acs/acs5/2021/B19013B`
- `acs/acs5/2024/B02001`
- `acs/acs5/2024/B15003`
- `acs/acs5/2024/B17001B`
- `acs/acs5/2024/B17001I`
- …and 2 more

**Peer comparisons:**
- *race_ethnicity* (nationwide size-matched): Columbus city, Georgia, Baton Rouge city, Louisiana, Washington city, District of Columbia, St. Louis city, Missouri, Greensboro city, North Carolina
- *education* (nationwide size-matched): Boston city, Massachusetts, Denver city, Colorado, Durham city, North Carolina, San Francisco city, California, Austin city, Texas
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q31 — L7 complex narrative

**Query:** Needs assessment for a youth workforce grant in Clayton County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 34 plan calls; 34 aggregated values; 7 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 42.7s · 34 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001607_998397`

**Response prose:**

> Clayton County, Georgia, with a total population of 298,924 in 2024, is characterized by significant demographic diversity, with approximately 69.4% of its residents identifying as Black or African American. The county faces substantial economic challenges, as evidenced by its 2024 poverty rate of 17.9%, which is 1.79 times the metro average and has risen by 9.2% since 2021. Concurrently, the median household income in Clayton County is $59,806, notably lower than the metro average at 0.65 times that figure, despite a 15.1% increase from 2021.
> >
> > In 2024, Clayton County had 80,080 children under 18. Data indicates an elevated general need, with a 'Needs assessment' ratio of 14.4%, which is somewhat above the metro level (1.22 times). The metric related to 'youth workforce grant' shows a ratio of 1.3%, roughly in line with the metro area (1.01 times), but has experienced a strong increase of 20.5% since 2021, suggesting growing demand or participation in such programs.
> >
> > These indicators collectively highlight a critical need for youth workforce development initiatives in Clayton County, where a substantial portion of the population experiences poverty and lower-than-average income, pointing to systemic barriers that a targeted grant could address.

**Key findings:**
- Clayton County's poverty rate of 17.9% in 2024 is significantly higher than the metro average (1.79 times) and has increased by 9.2% since 2021.
- The median household income in Clayton County was $59,806 in 2024, which is considerably lower than the metro average (0.65 times).
- The county has a substantial youth population, with 80,080 children under 18 in 2024.
- A general 'Needs assessment' indicator for Clayton County was elevated at 14.4% in 2024, exceeding the metro level by 1.22 times.
- Participation or eligibility related to 'youth workforce grants' in Clayton County increased by 20.5% from 2021 to 2024.
- Try next: What is the unemployment rate for young people here?
- Try next: How many young people here are not in school and not working?
- Try next: What is the high school graduation rate for youth in this area?

**Caveats:**
- The ACS 5-year estimates reflect data collected over a period and are not a single-year snapshot, which means they may lag behind the current grant period by up to three years.

**Citations** (17 rows, 7 unique tables):
- `acs/acs1/2021/B23001`
- `acs/acs1/2024/B01003`
- `acs/acs1/2024/B09001`
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B23001`
- `acs/acs5/2021/B01003`
- `acs/acs5/2021/B02001`
- `acs/acs5/2021/B09001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2024/B01003`
- …and 5 more

**Peer comparisons:**
- *age_structure* (within state 13): Gwinnett County, Georgia, Paulding County, Georgia, Houston County, Georgia, Muscogee County, Georgia, DeKalb County, Georgia
- *economic* (within state 13): Richmond County, Georgia, Henry County, Georgia, Chatham County, Georgia, Houston County, Georgia, Hall County, Georgia

---

### Q32 — L7 complex narrative

**Query:** First-time homebuyer gap for Black households in Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 20 plan calls; 20 aggregated values; 5 unique tables cited; 3 peer axises; 2 caveats
- **Runtime:** 41.2s · 20 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001700_680906`

**Response prose:**

> In 2024, Atlanta city's homeownership rate was 46.4%, significantly lower than Georgia's 66.3% and the national rate of 65.2%. This rate saw a modest 2.7% increase from 2021. The median home value in Atlanta city reached $439,600 in 2024, which is notably higher than the state median of $343,300 and the national median of $332,700. This represents a substantial 26.8% increase since 2021, indicating a rapidly appreciating housing market. While Atlanta's median household income of $85,652 in 2024 is roughly in line with state and national figures and increased by 23.8% since 2021, the elevated home values combined with a lower overall homeownership rate suggest significant affordability challenges. Given that Black residents comprise approximately 46.9% of Atlanta's population, these market dynamics likely create substantial barriers for Black households seeking to become first-time homebuyers.

**Key findings:**
- Atlanta's homeownership rate (46.4% in 2024) is considerably lower than both state (66.3%) and national (65.2%) averages.
- Median home values in Atlanta ($439,600 in 2024) are significantly higher than state and national medians, experiencing a strong 26.8% increase since 2021.
- Median household income in Atlanta ($85,652 in 2024) is comparable to state and national levels, with a strong 23.8% increase since 2021.
- The combination of lower overall homeownership, elevated home values, and a significant Black population suggests substantial affordability barriers for Black households.
- Try next: What is the homeownership rate for Black households here?
- Try next: What is the median household income for Black households?
- Try next: What share of Black households are rent-burdened in this area?

**Caveats:**
- The provided homeownership rate and income data are for the general population of Atlanta city and are not disaggregated by race.
- ACS homeownership data is based on householder race, and gaps at small-area levels can have wide margins of error.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25077`
- `acs/acs5/2021/B02001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25077`
- `acs/acs5/2021/B25081`
- `acs/acs5/2024/B02001`
- `acs/acs5/2024/B19013`
- `acs/acs5/2024/B25003`
- `acs/acs5/2024/B25077`
- …and 1 more

**Peer comparisons:**
- *race_ethnicity* (nationwide size-matched): Columbus city, Georgia, Baton Rouge city, Louisiana, Washington city, District of Columbia, St. Louis city, Missouri, Greensboro city, North Carolina
- *housing* (nationwide size-matched): Denver city, Colorado, Orlando city, Florida, Austin city, Texas, Raleigh city, North Carolina, Tampa city, Florida
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q33 — L7 complex narrative

**Query:** Climate resilience case for elderly residents in South Fulton.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 3 peer axises; 3 caveats
- **Runtime:** 50.6s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001751_856504`

**Response prose:**

> South Fulton faces significant vulnerabilities that could impact its elderly residents during climate events. In 2024, the area's poverty rate stood at 17.9%, which is substantially higher than the Atlanta metro area's 10.0% and the state's 12.6%. This high poverty rate, though showing a mild decrease of 9.3% since 2021, indicates a large segment of the population with limited resources to prepare for or recover from climate impacts. Mobility is another critical concern, with 14.0% of households in South Fulton lacking access to a vehicle in 2024. This figure is far above the metro area's 5.4% and the state's 6.0%, posing a significant challenge for evacuation or accessing resources during emergencies, particularly for elderly residents. While this rate has seen a strong decrease of 10.0% since 2021, it remains a major vulnerability. Data for 2024 also indicates an unweighted mean of 3,180 for disability status across tracts in South Fulton, a metric that has increased by 2.5% since 2021, highlighting the needs of a vulnerable population. Additionally, the unweighted mean for 'year structure built' across tracts was 1,539 in 2024, which, despite data interpretation limitations, may suggest the presence of older housing stock that could be more susceptible to climate impacts.

**Key findings:**
- South Fulton's 2024 poverty rate of 17.9% is substantially higher than the Atlanta metro area (10.0%) and the state (12.6%), indicating economic vulnerability.
- A significant 14.0% of households in South Fulton lacked a vehicle in 2024, far exceeding metro (5.4%) and state (6.0%) averages, posing mobility challenges for elderly residents during emergencies.
- The area has a notable population with disabilities, with an unweighted mean of 3,180 for disability status across tracts in 2024.
- Try next: What share of seniors here live alone?
- Try next: How many senior households here have broadband internet?
- Try next: What is the energy burden for low-income senior households here?

**Caveats:**
- Census data does not directly measure climate risk; it should be paired with hazard data from sources like NOAA, FEMA NFIP, or EJScreen.
- The reported values for 'disability status' and 'year structure built' are unweighted means across tracts, and their precise interpretation is limited without further context on the underlying tract-level metrics.
- The specific share of the population aged 65 and over could not be calculated from the provided data components.

**Citations** (12 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B25044`
- `acs/acs5/2021/B01001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B18101`
- `acs/acs5/2021/B25034`
- `acs/acs5/2021/B25044`
- `acs/acs5/2024/B01001`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B18101`
- `acs/acs5/2024/B25034`
- `acs/acs5/2024/B25044`

**Peer comparisons:**
- *age_structure* (within state 13): Cobb County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia
- *economic* (within state 13): Cobb County, Georgia, Gwinnett County, Georgia, DeKalb County, Georgia
- *disability* (within state 13): Gwinnett County, Georgia, Cobb County, Georgia, DeKalb County, Georgia

---

### Q34 — L7 complex narrative

**Query:** Kinship caregiving rates in Clayton County — grandparents raising grandchildren.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 35 plan calls; 35 aggregated values; 5 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 28.3s · 35 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001852_473654`

**Response prose:**

> In 2024, Clayton County had 8,836 grandparents living with grandchildren, a decrease of 18.1% from 2021. With 80,080 children under 18 in the county, this indicates that approximately 11.0% of children live in households with a grandparent, a rate higher than the Atlanta metro area (9.8%), Georgia (10.1%), and the United States (9.2%).

**Key findings:**
- In 2024, 8,836 grandparents in Clayton County were living with grandchildren, representing about 11.0% of children under 18.
- This kinship caregiving rate is higher than the Atlanta metro area (9.8%), Georgia (10.1%), and the United States (9.2%).
- The number of grandparents living with grandchildren in Clayton County decreased by 18.1% from 2021.
- Clayton County's poverty rate of 17.9% in 2024 is substantially higher than the metro average of 10.0% and has increased by 9.2% since 2021.
- The median household income in Clayton County was $59,806 in 2024, which is below the metro area's median of $92,344, despite a 15.1% increase from 2021.
- Try next: What is the child poverty rate in this area?
- Try next: What share of these grandparent caregivers are employed?
- Try next: What share of grandparent caregivers have a disability?

**Caveats:**
- The American Community Survey (ACS) identifies grandparents responsible for grandchildren, but formal foster placements are undercounted.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B09001`
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs5/2021/B09001`
- `acs/acs5/2021/B10054`
- `acs/acs5/2021/B10063`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2024/B09001`
- `acs/acs5/2024/B10054`
- `acs/acs5/2024/B10063`
- `acs/acs5/2024/B17001`
- …and 1 more

**Peer comparisons:**
- *housing* (within state 13): Muscogee County, Georgia, Gwinnett County, Georgia, Chatham County, Georgia, Bibb County, Georgia, Henry County, Georgia
- *family_structure* (within state 13): Muscogee County, Georgia, Bibb County, Georgia, Richmond County, Georgia, Chatham County, Georgia, DeKalb County, Georgia
- *commute_transit* (within state 13): Hall County, Georgia, Chatham County, Georgia, DeKalb County, Georgia, Gwinnett County, Georgia, Henry County, Georgia

---

### Q35 — L7 complex narrative

**Query:** Digital divide case for rural Georgia counties.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **5/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 6.5s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001852_473654`

**Response prose:**

> No data was returned for the query regarding a digital divide case for rural Georgia counties. This could be due to the query being too broad or specific for the available data, or the system was unable to identify specific rural counties to retrieve data for.

---

### Q36 — L8 multi-concept

**Query:** Food access indicators across Southwest Atlanta, Clayton County, and the metro.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 55 plan calls; 55 aggregated values; 5 unique tables cited; 3 peer axises; 2 caveats
- **Runtime:** 41.0s · 55 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_001947_282682`

**Response prose:**

> In 2024, Southwest Atlanta, Clayton County, and the Atlanta Metro Way of the Cross exhibit significant food access challenges, characterized by elevated poverty, high SNAP participation, and limited vehicle access compared to the broader Atlanta metro area.
> >
> > Southwest Atlanta faces particularly acute conditions. Its SNAP participation rate is 27.5%, far above the metro level (3.38 times), and its poverty rate is 22.3%, also far exceeding the metro average (2.22 times). A substantial 18.6% of households are without a vehicle, far above the metro rate (3.47 times), indicating significant barriers to accessing grocery stores. The median household income is $57,568, far below the metro's $92,344, and its child poverty rate of 6.8% is far above the metro level (1.95 times).
> >
> > Clayton County and the Atlanta Metro Way of the Cross also show high indicators of food insecurity. Clayton County's SNAP participation rate is 19.3% (2.37 times the metro level), its poverty rate is 17.9% (1.79 times the metro level), and its child poverty rate is 7.3% (2.09 times the metro level). The Atlanta Metro Way of the Cross has a SNAP participation rate of 17.9% (2.19 times the metro level) and 8.1% of households without a vehicle (1.50 times the metro level). Both areas have median household incomes significantly below the metro average, with Clayton County at $59,806 and the Atlanta Metro Way of the Cross at $40,690. The child poverty rate in the Atlanta Metro Way of the Cross is reported as 0.0% in 2024, a strong decrease from 2021.

**Key findings:**
- In 2024, Southwest Atlanta's SNAP participation rate is 27.5%, its poverty rate is 22.3%, and 18.6% of households are without a vehicle, all significantly higher than the Atlanta metro average.
- Clayton County's 2024 SNAP participation rate is 19.3%, poverty rate is 17.9%, and child poverty rate is 7.3%, all substantially above metro levels.
- The Atlanta Metro Way of the Cross shows a 2024 SNAP participation rate of 17.9% and 8.1% of households without a vehicle, both above metro averages, alongside a median household income of $40,690, far below the metro.
- All three geographies have median household incomes significantly below the metro average in 2024.
- Try next: How many residents here live in a food desert?
- Try next: What share of households on SNAP have children under 18?
- Try next: What are the rates of diet-related health conditions here?

**Caveats:**
- Census data approximates food-access conditions; direct food-insecurity measures come from USDA ERS / Feeding America.
- The median household income for Southwest Atlanta is an unweighted mean across tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.

**Citations** (14 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B22003`
- `acs/acs1/2024/B25044`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B17020`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B22003`
- `acs/acs5/2021/B25044`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B17020`
- `acs/acs5/2024/B19013`
- …and 2 more

**Peer comparisons:**
- *food_benefits* (nationwide size-matched): Columbus city, Ohio, Kansas City city, Kansas, Killeen city, Texas, Columbus city, Georgia, Chattanooga city, Tennessee
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia
- *commute_transit* (nationwide size-matched): Minneapolis city, Minnesota, Portland city, Oregon, Denver city, Colorado, Alexandria city, Virginia, Miami city, Florida

---

### Q37 — L8 multi-concept

**Query:** Housing instability comparison between Buckhead and East Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 45 plan calls; 45 aggregated values; 5 unique tables cited; 2 peer axises; 2 caveats
- **Runtime:** 22.0s · 45 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002038_235422`

**Response prose:**

> In 2024, Buckhead and East Atlanta exhibit contrasting housing stability profiles. Buckhead, with a median household income of $139,871, which is substantially above the Atlanta metro level (1.51x), shows a low poverty rate of 7.3%, far below the Fulton County average (0.54x). Its rent burden is 47.1%, roughly in line with the metro level, but has risen by 19.5% since 2021. The homeownership rate in Buckhead is 50.6%, below the metro level (0.76x).
> >
> > East Atlanta faces significantly higher housing instability. Its median household income is $86,167, roughly in line with the metro level, but has seen a notable decrease of 34.2% since 2021. The poverty rate is 13.7%, somewhat above the metro level (1.37x), and has risen sharply by 70.0% since 2021. Rent burden in East Atlanta is 68.8%, elevated and somewhat above the metro level (1.29x), with a substantial increase of 74.6% since 2021. The homeownership rate is 65.6%, roughly in line with the metro level, and has risen by 29.6% since 2021.
> >
> > Overall, East Atlanta demonstrates greater housing instability, characterized by a higher rent burden, higher poverty rate, and a significant decline in median household income, contrasting sharply with Buckhead's more affluent and stable housing environment.

**Key findings:**
- In 2024, East Atlanta's rent burden is 68.8%, significantly higher than Buckhead's 47.1%.
- East Atlanta's poverty rate is 13.7%, nearly double Buckhead's 7.3%.
- Buckhead's median household income is $139,871, substantially higher than East Atlanta's $86,167.
- East Atlanta has experienced notable negative trends since 2021, including a 74.6% increase in rent burden, a 70.0% increase in poverty rate, and a 34.2% decrease in median household income.
- Try next: How has the racial composition of households changed recently?
- Try next: What share of low-income households are rent-burdened here?
- Try next: What is the share of renter-occupied housing units here?

**Caveats:**
- Median gross rent and median household income values for Buckhead and East Atlanta are unweighted means across tracts; true sub-county medians require population-weighted interpolation against bracketed-income tables.
- Rent and cost-burden data reflect 5-year ACS averages; they lag current conditions by up to 3 years.

**Citations** (15 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B25003`
- `acs/acs1/2024/B25064`
- `acs/acs1/2024/B25070`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B25003`
- `acs/acs5/2021/B25064`
- `acs/acs5/2021/B25070`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B19013`
- …and 3 more

**Peer comparisons:**
- *housing* (nationwide size-matched): Peachtree Heights West, Brookwood, North Buckhead, Old Fourth Ward, Lindridge/Martin Manor
- *economic* (nationwide size-matched): Grant Park, Kirkwood, Reynoldstown, Old Fourth Ward, Home Park

---

### Q38 — L8 multi-concept

**Query:** Theory of change for a community health program in Fulton County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **6/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 1.7s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002038_235422`

**Response prose:**

> I don't have Census data on that topic. The Census Bureau provides data, not theories of change or program planning frameworks. I can help with demographic, economic, or health insurance data that might inform your program.
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q39 — L8 multi-concept

**Query:** Broadband expansion grant for a rural high-poverty Georgia area.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **5/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 0.0s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002038_235422`

**Response prose:**

> I don't have Census data on that topic. contains out-of-scope token 'band' — Census does not track this subject
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q40 — L8 multi-concept

**Query:** Environmental justice indicators for the Westside of Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **6/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 2.9s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002038_235422`

**Response prose:**

> I don't have Census data on that topic. The Census Bureau does not collect environmental quality data like pollution or air quality, which are key components of environmental justice indicators. I can provide demographic and socioeconomic data relevant to environmental justice analyses, such as poverty, race, and income, for the Westside of Atlanta.
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q41 — L9 nuanced edge

**Query:** Substance use prevention program — demographic risk indicators for DeKalb.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 30 plan calls; 30 aggregated values; 6 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 46.0s · 30 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002144_849646`

**Response prose:**

> In 2024, DeKalb County exhibits several demographic risk indicators relevant for substance use prevention programs. The unemployment rate was 6.4%, which is substantially above the metro level (1.41 times) and higher than both the state (1.36 times) and national (1.22 times) rates. This rate has fallen by 4.5% since 2021. The poverty rate in DeKalb County was 13.7%, elevated compared to the metro area (1.36 times) and slightly above state and national averages. This rate has remained flat since 2021. Median household income was $80,644, roughly in line with state and national figures, but somewhat below the metro area, representing a strong increase of 16.2% since 2021. DeKalb County's population with a disability was 760,222, a figure flagged as far below the metro average (0.12 times) and considered outside the expected range. This count represents a 0.9% increase since 2021. The county's population in 2024 consisted of 363,095 males and 402,256 females.

**Key findings:**
- DeKalb County's 2024 unemployment rate was 6.4%, substantially higher than metro, state, and national averages, though it decreased by 4.5% since 2021.
- The 2024 poverty rate was 13.7%, elevated compared to the metro area and slightly above state and national averages.
- Median household income in 2024 was $80,644, showing a strong 16.2% increase since 2021, aligning with state and national figures.
- The population with a disability was 760,222 in 2024, a figure flagged as far below the metro average.
- Try next: What share of adults here report frequent mental distress?
- Try next: What is the high school dropout rate in this area?
- Try next: What share of the population here lacks health insurance?

**Caveats:**
- The American Community Survey (ACS) data does not directly measure substance use or overdose rates. For actual rates, refer to sources like CDC WONDER, state Medicaid data, or SAMHSA.

**Citations** (15 rows, 6 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B23025`
- `acs/acs1/subject/2021/S0101`
- `acs/acs1/subject/2024/S0101`
- `acs/acs5/2021/B01001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B18101`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B23025`
- `acs/acs5/2024/B01001`
- `acs/acs5/2024/B17001`
- …and 3 more

**Peer comparisons:**
- *disability* (within state 13): Paulding County, Georgia, Coweta County, Georgia, Hall County, Georgia, Henry County, Georgia, Houston County, Georgia
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia

---

### Q42 — L9 nuanced edge

**Query:** Homelessness risk indicators for Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 24 plan calls; 24 aggregated values; 5 unique tables cited; 2 peer axises; 1 caveat
- **Runtime:** 42.1s · 24 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002240_886132`

**Response prose:**

> In 2024, Atlanta city's overall homelessness risk indicators were 60.7%, roughly in line with the state level (55.3%) and national level (57.7%). This represents a mild decrease of 9.2% from 66.9% in 2021. However, several underlying factors suggest ongoing housing instability. The severe rent burden in Atlanta city was 26.8% in 2024, indicating that over a quarter of households spend more than 50% of their income on rent. This figure is roughly in line with the state (24.4%) and national (24.1%) averages, but has seen a strong increase of 10.1% since 2021. The median gross rent in Atlanta city was $1,711 in 2024, significantly higher than the state median of $1,506 and the national median of $1,413. This represents a notable shift, with median gross rent increasing by 27.5% since 2021.
> >
> > Furthermore, Atlanta city's poverty rate stood at 16.9% in 2024, which is elevated compared to the state (12.6%) and national (12.5%) rates, despite a mild decrease of 8.6% from 2021. The proportion of single-parent households, often a marker of precarious family structure, was 18.2% in 2024. This is also elevated compared to Georgia (14.5%) and the US (13.0%), though it has mildly fallen by 8.1% since 2021. These combined factors highlight significant challenges for many Atlanta residents in maintaining stable housing.

**Key findings:**
- Atlanta city's homelessness risk indicators were 60.7% in 2024, a mild decrease from 2021 but still comparable to state and national levels.
- Severe rent burden affected 26.8% of households in 2024, a strong increase of 10.1% since 2021.
- Median gross rent in Atlanta city was $1,711 in 2024, significantly higher than state and national figures, and increased by 27.5% since 2021.
- The poverty rate (16.9%) and proportion of single-parent households (18.2%) in Atlanta city were elevated compared to state and national averages in 2024.
- Try next: What are the recent eviction rates in this area?
- Try next: What share of the population here lives in deep poverty?
- Try next: How does rent burden vary across income brackets here?

**Caveats:**
- The American Community Survey (ACS) data does not measure literal homelessness; it provides indicators of populations at risk. Official counts of homelessness are typically sourced from HUD's Point-in-Time (PIT) counts and state Homeless Management Information Systems (HMIS).

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2021/C17019`
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B25064`
- `acs/acs1/2024/B25070`
- `acs/acs1/2024/C17019`
- `acs/acs5/2021/B11003`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B25064`
- `acs/acs5/2021/B25070`
- `acs/acs5/2024/B11003`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B25064`
- …and 1 more

**Peer comparisons:**
- *housing* (nationwide size-matched): Denver city, Colorado, Orlando city, Florida, Austin city, Texas, Raleigh city, North Carolina, Tampa city, Florida
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q43 — L9 nuanced edge

**Query:** Language access case for the city of Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **5/10** — 0 plan calls; 0 aggregated values
- **Runtime:** 14.2s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002240_886132`

**Response prose:**

> I am sorry, but I do not have the necessary data to construct a language access case for the city of Atlanta. The requested information on Limited English Proficiency (LEP) share, language mix, and foreign-born context was not returned.

---

### Q44 — L9 nuanced edge

**Query:** Disability services baseline for Atlanta.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 20 plan calls; 20 aggregated values; 5 unique tables cited; 2 peer axises; 2 caveats
- **Runtime:** 49.5s · 20 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002357_160338`

**Response prose:**

> In 2024, the population for which disability status was measured in Atlanta city was 497,595, representing an increase of 2.8% since 2021. Within this measured population, 472,939 individuals were assessed for ambulatory difficulty, a 3.3% rise from 2021, and 413,606 for independent living difficulty, marking a 3.8% increase over the same period. While specific prevalence rates for disability are not directly available in the provided data, these figures indicate the scale of the population for whom disability services may be relevant. The economic context in Atlanta city further highlights areas of need. In 2024, the poverty rate was 16.9%, which is somewhat above the state level of 12.6% and the national rate of 12.5%. While this rate has seen a positive trend, decreasing by 8.6% since 2021, it remains elevated. The median household income in Atlanta city was $85,652 in 2024, roughly in line with the state's median of $79,991 and the national median of $80,734. This income level has shown strong growth, increasing by 23.8% since 2021.

**Key findings:**
- In 2024, the population measured for disability status in Atlanta city was 497,595, increasing by 2.8% since 2021.
- The population measured for ambulatory difficulty was 472,939 (up 3.3% from 2021), and for independent living difficulty was 413,606 (up 3.8% from 2021).
- Atlanta city's poverty rate was 16.9% in 2024, which is elevated compared to state and national averages, despite an 8.6% decrease since 2021.
- Median household income in Atlanta city was $85,652 in 2024, showing strong growth of 23.8% since 2021.
- Try next: What's the prevalence of cognitive, vision, or hearing difficulties here?
- Try next: What's the employment rate for people with disabilities in this area?
- Try next: How many people with disabilities here lack access to a vehicle?

**Caveats:**
- ACS disability items are self-reported and capture 6 functional domains only; clinical definitions differ.
- Specific prevalence rates for disability and functional limitations are not directly available in the provided data.

**Citations** (12 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B18101`
- `acs/acs5/2021/B18105`
- `acs/acs5/2021/B18107`
- `acs/acs5/2021/B19013`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B18101`
- `acs/acs5/2024/B18105`
- `acs/acs5/2024/B18107`
- `acs/acs5/2024/B19013`

**Peer comparisons:**
- *disability* (nationwide size-matched): Washington city, District of Columbia, Nashville-Davidson metropolitan government (balance), Tennessee, Aurora city, Colorado, Columbus city, Ohio, Riverside city, California
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q45 — L9 nuanced edge

**Query:** Mental health services case for DeKalb County.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 25 plan calls; 25 aggregated values; 5 unique tables cited; 3 peer axises; 1 caveat
- **Runtime:** 43.1s · 25 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002456_700260`

**Response prose:**

> In 2024, DeKalb County faces significant challenges in mental health service access, with its uninsured rate standing at 12.7%, which is far above the national average (1.51 times). The county's poverty rate is 13.7%, somewhat above the metro level (1.36 times). While the number of residents with a disability (760,222 people) and seniors living alone (173,754 people) are substantial, these figures are far below the metro area's proportional levels (0.12 times for both). The median household income in DeKalb County was $80,644 in 2024, showing a strong increase of 16.2% since 2021.

**Key findings:**
- DeKalb County's uninsured rate of 12.7% in 2024 is significantly higher than the national average.
- The county's poverty rate of 13.7% in 2024 is elevated compared to the metro area.
- In 2024, 760,222 residents live with a disability and 173,754 seniors live alone, indicating populations that may require mental health support.
- Median household income in DeKalb County rose to $80,644 in 2024, a 16.2% increase from 2021.
- Try next: What share of people here report cognitive difficulty?
- Try next: What is the child poverty rate in this area?
- Try next: How many households here don't have a vehicle?

**Caveats:**
- The American Community Survey (ACS) does not directly measure mental health; proxies such as disability, isolation, and uninsured status are used to approximate service demand.

**Citations** (13 rows, 5 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs1/2024/B19013`
- `acs/acs1/2024/B27010`
- `acs/acs5/2021/B09010`
- `acs/acs5/2021/B17001`
- `acs/acs5/2021/B18101`
- `acs/acs5/2021/B19013`
- `acs/acs5/2021/B27010`
- `acs/acs5/2024/B09010`
- `acs/acs5/2024/B17001`
- `acs/acs5/2024/B18101`
- `acs/acs5/2024/B19013`
- …and 1 more

**Peer comparisons:**
- *disability* (within state 13): Paulding County, Georgia, Coweta County, Georgia, Hall County, Georgia, Henry County, Georgia, Houston County, Georgia
- *economic* (within state 13): Chatham County, Georgia, Gwinnett County, Georgia, Cobb County, Georgia, Hall County, Georgia, Houston County, Georgia
- *age_structure* (within state 13): Muscogee County, Georgia, Cobb County, Georgia, Houston County, Georgia, Columbia County, Georgia, Paulding County, Georgia

---

### Q46 — L10 scope gate

**Query:** What's the weather forecast for Atlanta this weekend?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **10/10** — scope gate correctly rejected an out-of-Census query with a polite, topic-specific refusal
- **Runtime:** 0.0s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002456_700260`

**Response prose:**

> I don't have Census data on that topic. contains out-of-scope token 'weather' — Census does not track this subject
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q47 — L10 scope gate

**Query:** How many Waffle Houses are in Fulton County?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **10/10** — scope gate correctly rejected an out-of-Census query with a polite, topic-specific refusal
- **Runtime:** 1.3s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002456_700260`

**Response prose:**

> I don't have Census data on that topic. The Census doesn't track individual restaurants or franchise counts. I can report total business establishments by industry (via the Economic Census) for a general food- services count if that helps.
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q48 — L10 scope gate

**Query:** How many jet skis are in Northeast Atlanta?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **10/10** — scope gate correctly rejected an out-of-Census query with a polite, topic-specific refusal
- **Runtime:** 0.0s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002456_700260`

**Response prose:**

> I don't have Census data on that topic. contains out-of-scope token 'jet ski' — Census does not track this subject
> >
> > I can help with demographic, economic, housing, transportation, health-insurance, and education indicators for US geographies.

---

### Q49 — L10 exclusion

**Query:** I'm writing a grant for more school buses in Atlanta excluding the Buckhead area.

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Mostly** — prose present but brief.
- **Rating:** **7/10** — exclusion caveat surfaced; but produced 0 plan calls (extraction dropped the compound then extracted no primary geo)
- **Runtime:** 11.9s · 0 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002456_700260`

**Response prose:**

> No specific Census data directly related to school bus needs or transportation infrastructure was found for Atlanta, excluding the Buckhead area, in the latest available datasets. Census data primarily focuses on demographic, economic, and housing characteristics, rather than specific operational needs like school bus fleets or student transportation challenges.

**Caveats:**
- Geographic exclusions are on the roadmap but not yet supported — I treated the query as the whole primary area without subtracting: 'Atlanta' excluding 'Buckhead area'. The numbers below therefore include the excluded sub-area.

**Peer comparisons:**
- *economic* (nationwide size-matched): Boston city, Massachusetts, Tampa city, Florida, Oakland city, California, Jersey City city, New Jersey, Washington city, District of Columbia

---

### Q50 — L10 relative geo

**Query:** What's the poverty rate near the Atlanta Beltline?

- **Completed:** Yes
- **Returned something:** Yes
- **Informative:** **Yes** — substantive prose with key findings.
- **Rating:** **10/10** — 6 plan calls; 6 aggregated values; 1 unique tables cited; 1 peer axis
- **Runtime:** 10.4s · 6 plan calls
- **Dump folder:** `/home/kmcalist/KaggleComps/GACensusScraper/agentic-census-chatbot/derived_data/data_20260423_002643_058762`

**Response prose:**

> In 2024, the poverty rate for the BeltLine Eastside Trail was 11.2%. This rate is roughly in line with the Atlanta-Sandy Springs-Alpharetta, GA Metro Area's rate of 10.0% and lower than Fulton County's rate of 12.3%, Georgia's rate of 12.6%, and the national rate of 12.5%. The poverty rate in this area has shown a falling trend, decreasing by 9.2% from 12.3% in 2021.

**Key findings:**
- The poverty rate for the BeltLine Eastside Trail was 11.2% in 2024.
- This rate is similar to the metro area's poverty rate (10.0%) and lower than the county, state, and national averages.
- The poverty rate in the area has decreased by 9.2% since 2021.
- Try next: How has the poverty rate changed here recently?
- Try next: How does the poverty rate here compare to the metro average?
- Try next: What's the child poverty rate in this area?

**Citations** (3 rows, 1 unique tables):
- `acs/acs1/2024/B17001`
- `acs/acs5/2021/B17001`
- `acs/acs5/2024/B17001`

**Peer comparisons:**
- *economic* (nationwide size-matched): Pine Hills, Virginia Highland, Grant Park, Morningside/Lenox Park, Kirkwood

---
